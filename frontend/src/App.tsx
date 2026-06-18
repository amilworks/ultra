import {
  Suspense,
  lazy,
  memo,
  type CSSProperties,
  type ComponentType,
  type MutableRefObject,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";
import {
  ChatContainerContent,
  ChatContainerRoot,
  ChatContainerScrollAnchor,
  FileUpload,
  FileUploadTrigger,
  Loader,
  MarkdownResponseStream,
  Message,
  MessageAction,
  MessageActions,
  MessageContent,
  PromptInput,
  PromptInputAction,
  PromptInputActions,
  PromptInputTextarea,
  ScrollButton,
  SystemMessage,
  ThinkingBar,
} from "./components/prompt-kit";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Input } from "@/components/ui/input";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarInput,
  SidebarMenuAction,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { useBreakpoint } from "@/hooks/use-breakpoint";
import { cn } from "@/lib/utils";
import { ApiClient, ApiError, UploadPausedError, type UploadProgressEvent } from "./lib/api";
import { buildNavUrl, navStateKey, parseNavFromSearch, type NavState } from "./lib/navUrl";
import {
  DEFAULT_API_BASE_URL,
  DEFAULT_API_KEY,
  DEFAULT_BISQUE_BROWSER_URL,
} from "./lib/config";
import { buildBisqueThumbnailUrl } from "./lib/bisquePreview";
import { formatBytes, formatTokens } from "./lib/format";
import {
  buildBisqueNavLinks,
  inferBisqueRootFromUrl,
  type BisqueNavLinks,
} from "./features/auth/bisqueNavigation";
import {
  createAdminOrganization,
  createAdminUser,
  deleteAdminUser,
  loadAdminOrganizations,
  loadAdminIssues,
  loadAdminOverview,
  loadAdminRuns,
  loadAdminUsers,
  updateAdminUserStatus,
} from "./features/admin/client";
import {
  listRunArtifacts,
  listRunEvents,
  listSessionConversations,
} from "./features/chat/client";
import {
  buildScientificResultGroups,
} from "./features/chat/scientific-results";
import {
  isHydratableRunArtifactVisual,
  rewriteArtifactMarkdownImageUrls,
  shouldHydrateRunArtifacts,
} from "./features/chat/run-artifact-hydration";
import {
  isLegacyRunLookup404Content,
  latestRunEventSequence,
  shouldDropHydratedLegacy404Message,
  shouldRecoverRunResultMessage,
} from "./features/chat/run-recovery";
import { extractRunTokenUsage } from "./features/chat/token-usage";
import {
  findReusableBlankDraftConversation,
  shouldShowConversationInHistory,
  shouldPersistConversationSnapshot,
} from "./features/chat/conversation-draft";
import {
  fallbackConversationTitleFromText,
  resolveConversationTitle,
} from "./features/chat/conversation-title";
import {
  prependResolvedConversation,
  shouldKeepOptimisticConversationAfterHydration,
} from "./features/chat/stale-conversation";
import {
  createBulkResourceShareGrants as createBulkResourceShareGrantsRequest,
  createResourceCollectionShareGrants as createResourceCollectionShareGrantsRequest,
  createResourceShareGrant as createResourceShareGrantRequest,
  deleteBulkResources as deleteBulkResourcesRequest,
  deleteResourceCollection as deleteResourceCollectionRequest,
  loadComposerResources,
  loadLibraryResources,
  loadResourceFolders,
  loadResourceFolderResources,
  loadResourceShareGrants as loadResourceShareGrantsRequest,
  removeResourceFromCollection as removeResourceFromCollectionRequest,
  renameResource as renameResourceRequest,
  renameResourceCollection as renameResourceCollectionRequest,
  restoreBulkResources as restoreBulkResourcesRequest,
  restoreResourceCollection as restoreResourceCollectionRequest,
  restoreResource as restoreResourceRequest,
  revokeResourceShareGrant as revokeResourceShareGrantRequest,
} from "./features/resources/client";
import {
  createResourceUploadQueueStore,
  hydrateResourceUploadProgressFromQueueStore,
  persistResourceUploadProgressEvent,
} from "./features/resources/uploadQueueStore";
import {
  createResourceUploadProgressFrameBatcher,
  mergeResourceUploadProgress,
} from "./features/resources/uploadProgressBatcher";
import {
  DEFAULT_THINKING_TEXT,
  getPhaseThinkingText,
  getToolStatusThinkingText,
} from "./lib/runStepCopy";
import { useLocalStorageState } from "./lib/useLocalStorageState";
import { UserTokenUsagePanel } from "./components/UserTokenUsagePanel";
import type {
  AdminCreateOrganizationRequest,
  AdminCreateUserRequest,
  AdminIssueRecord,
  AdminOrganization,
  AdminOverviewResponse,
  AdminRunRecord,
  AdminUserStatus,
  AdminUserSummary,
  AssistantContract,
  ArtifactRecord,
  BisqueAuthSessionResponse,
  ChatResponse,
  ChatMessage,
  ConversationRecord,
  CurrentUserProfile,
  ProgressEvent,
  ResourceComputationSuggestion,
  ResourceCollectionRecord,
  ResourceRecord,
  ResourceShareGrantRecord,
  RunEvent,
  SelectionContext,
  TokenUsageResponse,
  UploadedFileRecord,
} from "./types";
import { BisqueMarkIcon } from "./components/icons/BisqueMarkIcon";
import { LensSidebarIcon } from "./components/icons/LensSidebarIcon";
import { RunningStatusPill } from "./components/chat/RunningStatusPill";
import type {
  ComposerWorkflowDefinition,
  ComposerWorkflowId,
  ComposerWorkflowPresetState,
} from "./components/chat/composer-workflows";
import type {
  MegasegFileInsight,
  PrairieImageAnalysis,
  ResearchDigestData,
  ResearchDigestMeasurementRow,
  ResearchDigestStatisticRow,
  ScientificFigureCard,
  ToolCardImage,
  ToolCardMetric,
  ToolDetectionBox,
  ToolDownloadRow,
  ToolImageHoverDetails,
  ToolResourceRow,
  ToolResultCard,
  YoloFigureCard,
  YoloFigureClassCount,
} from "./components/chat/ToolResultCards";
import type {
  ResourceCollectionAddSelectionRequest,
  ResourceCollectionSelectionRequest,
  ResourceKindFilter,
  ResourceShareGrantRequest,
  ResourceSharingFilter,
  ResourceStatusFilter,
  ResourceUploadProgress,
  ResourceUploadReselectionContext,
  ResourceSourceFilter,
} from "./components/ResourceBrowser";
import {
  ArrowUp,
  Check,
  ChevronDown,
  Copy,
  Database,
  FolderOpen,
  ImageIcon,
  Images,
  Laptop,
  Link2,
  LogOut,
  MessageCircle,
  Moon,
  MoreHorizontal,
  Pencil,
  Plus,
  PlusIcon,
  Settings,
  Shield,
  Square,
  Sun,
  Table2,
  ThumbsDown,
  ThumbsUp,
  Trash,
  X,
} from "lucide-react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import type * as Sonner from "sonner";

type SonnerModule = typeof Sonner;
type ToastSuccessOptions = Parameters<SonnerModule["toast"]["success"]>[1];

type UiRole = "user" | "assistant";
type ThemePreference = "system" | "light" | "dark";
type AuthMode = "bisque" | "guest" | "workos";
type AuthProvider = "local" | "workos";
type AuthStatus = "checking" | "authenticated" | "unauthenticated";
type ActivePanel = "chat" | "resources" | "admin" | "training" | "scientific-viewer";
type ComposerIntelligenceMode = "high" | "pro";
type BisqueResourceCounts = {
  image: number;
  dataset: number;
  table: number;
};
type BisqueResourceCountsState = {
  requestKey: string;
  counts: BisqueResourceCounts | null;
};

const MISSING_REQUESTED_CONVERSATION_MESSAGE =
  "Requested chat was not found. Opened the latest available conversation instead.";

export const shouldShowAppShellBanner = (
  activePanel: ActivePanel,
  message: string | null
): boolean => {
  if (!message) {
    return false;
  }
  return activePanel === "chat" || message !== MISSING_REQUESTED_CONVERSATION_MESSAGE;
};

const queueEffectUpdate = (callback: () => void): (() => void) => {
  let cancelled = false;
  window.queueMicrotask(() => {
    if (!cancelled) {
      callback();
    }
  });
  return () => {
    cancelled = true;
  };
};

const readAuthErrorFromLocation = (): string | null => {
  if (typeof window === "undefined") {
    return null;
  }
  const value = new URLSearchParams(window.location.search).get("auth_error");
  return value && value.trim().length > 0 ? value.trim() : null;
};

const clearAuthErrorFromLocation = (): void => {
  if (typeof window === "undefined") {
    return;
  }
  const params = new URLSearchParams(window.location.search);
  if (!params.has("auth_error")) {
    return;
  }
  params.delete("auth_error");
  const nextQuery = params.toString();
  const nextUrl = `${window.location.pathname}${nextQuery ? `?${nextQuery}` : ""}${window.location.hash}`;
  window.history.replaceState({}, "", nextUrl);
};

type WorkOSRedirectScreenProps = {
  checking: boolean;
  loading: boolean;
  errorMessage?: string | null;
  statusMessage?: string | null;
  onRetry: () => Promise<void> | void;
};

function WorkOSRedirectScreen({
  checking,
  loading,
  errorMessage,
  statusMessage,
  onRetry,
}: WorkOSRedirectScreenProps) {
  const title = checking
    ? "Checking your session"
    : statusMessage
      ? "Account not yet available"
      : errorMessage
        ? "Unable to open WorkOS"
        : "Opening WorkOS sign in";
  const message =
    statusMessage ||
    errorMessage ||
    "Taking you directly to the BisQue Ultra sign-in page.";
  const canRetry = !checking && !loading && Boolean(errorMessage || statusMessage);

  return (
    <main className="grid min-h-svh place-items-center bg-background p-6 text-foreground">
      <section
        className="grid w-full max-w-sm gap-4 rounded-2xl border bg-card p-6 text-center shadow-sm"
        aria-live="polite"
      >
        <div className="mx-auto flex size-11 items-center justify-center rounded-xl bg-muted text-muted-foreground">
          <BisqueMarkIcon className="size-5" />
        </div>
        <div className="grid gap-1">
          <h1 className="text-base font-semibold tracking-normal">{title}</h1>
          <p className="text-sm leading-6 text-muted-foreground">{message}</p>
        </div>
        {canRetry ? (
          <Button type="button" className="h-9 rounded-xl" onClick={() => void onRetry()}>
            Try again
          </Button>
        ) : null}
      </section>
    </main>
  );
}

function AuthScreenLoadingFallback() {
  return (
    <main className="grid min-h-svh place-items-center bg-background p-6 text-foreground">
      <section className="grid w-full max-w-sm gap-3 rounded-2xl border bg-card p-6 text-center shadow-sm">
        <Loader className="mx-auto size-5 animate-spin text-muted-foreground" />
        <h1 className="text-xl font-semibold tracking-tight">Opening sign in</h1>
        <p className="text-sm text-muted-foreground">
          Preparing BisQue Ultra.
        </p>
      </section>
    </main>
  );
}

function UploadFlightChip({
  inFlightCount,
  onOpen,
}: {
  inFlightCount: number;
  onOpen: () => void;
}) {
  if (inFlightCount <= 0) {
    return null;
  }
  return (
    <button
      type="button"
      className="upload-flight-chip"
      onClick={onOpen}
      aria-label={`Uploading ${inFlightCount} ${
        inFlightCount === 1 ? "file" : "files"
      }. Open Resources.`}
    >
      <Loader className="upload-flight-chip-spinner" aria-hidden="true" />
      <span className="upload-flight-chip-label">Uploading {inFlightCount}...</span>
    </button>
  );
}

function DeferredToaster({ theme }: { theme: "light" | "dark" }) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const show = () => setReady(true);
    if (typeof window.requestIdleCallback === "function") {
      const idleId = window.requestIdleCallback(show, { timeout: 3_000 });
      return () => window.cancelIdleCallback(idleId);
    }

    const timeoutId = window.setTimeout(show, 1_200);
    return () => window.clearTimeout(timeoutId);
  }, []);

  if (!ready) {
    return null;
  }

  return (
    <Suspense fallback={null}>
      <LazyToaster
        theme={theme}
        richColors
        position="bottom-right"
        toastOptions={{ duration: 4200 }}
      />
    </Suspense>
  );
}

type RunImageArtifact = {
  path: string;
  url: string;
  title: string;
  sourceName: string;
  sourcePath?: string;
  previewable: boolean;
  downloadUrl?: string;
  linkedFileId?: string | null;
  resultGroupId?: string | null;
};

const formatBisqueShortcutLabel = (
  count: number | null | undefined,
  singular: string,
  plural: string
): string => {
  if (!Number.isFinite(count)) {
    return plural;
  }
  const normalizedCount = Math.max(0, Math.floor(Number(count)));
  return `${normalizedCount.toLocaleString()} ${
    normalizedCount === 1 ? singular : plural
  }`;
};

const lazyNamed = <TModule extends Record<string, unknown>>(
  loader: () => Promise<TModule>,
  exportName: keyof TModule
) =>
  lazy(async () => {
    const module = await loader();
    return {
      default: module[exportName] as ComponentType<any>,
    };
  });

const loadUploadViewerSheetModule = () => import("./components/UploadViewerSheet");
const loadAdminConsoleModule = () => import("./components/AdminConsole");
const loadAppSettingsDialogModule = () => import("./components/AppSettingsDialog");
const loadAuthScreenModule = () => import("./components/auth/AuthScreen");
const loadTrainingDashboardModule = () => import("./components/TrainingDashboard");
const loadScientificViewerPageModule = () => import("./components/ScientificViewerPage");
const loadResourceBrowserModule = () => import("./components/ResourceBrowser");
const loadComposerSlashMenuModule = () => import("./components/chat/ComposerSlashMenu");
const loadChatRunStepsModule = () => import("./components/chat/ChatRunSteps");
const loadInlineDataQuickPreviewModule = () =>
  import("./components/chat/InlineDataQuickPreview");
const loadProModeDevTraceModule = () => import("./components/chat/ProModeDevTrace");
const loadToolResultCardsModule = () => import("./components/chat/ToolResultCards");
const loadComposerWorkflowsModule = () => import("./components/chat/composer-workflows");

type ComposerWorkflowsModule = Awaited<
  ReturnType<typeof loadComposerWorkflowsModule>
>;

let composerWorkflowsModulePromise: Promise<ComposerWorkflowsModule> | null = null;

const loadComposerWorkflows = (): Promise<ComposerWorkflowsModule> => {
  composerWorkflowsModulePromise ??= loadComposerWorkflowsModule().catch(
    (error: unknown) => {
      composerWorkflowsModulePromise = null;
      throw error;
    }
  );
  return composerWorkflowsModulePromise;
};

let sonnerModulePromise: Promise<SonnerModule> | null = null;

const loadSonnerModule = (): Promise<SonnerModule> => {
  sonnerModulePromise ??= import("sonner").catch((error: unknown) => {
    sonnerModulePromise = null;
    throw error;
  });
  return sonnerModulePromise;
};

const showSuccessToast = (
  message: string,
  options?: ToastSuccessOptions
): void => {
  void loadSonnerModule().then(({ toast }) => {
    toast.success(message, options);
  });
};

type ToastErrorOptions = Parameters<SonnerModule["toast"]["error"]>[1];

const showErrorToast = (
  message: string,
  options?: ToastErrorOptions
): void => {
  void loadSonnerModule().then(({ toast }) => {
    toast.error(message, options);
  });
};

const RESOURCE_UPLOAD_IN_FLIGHT_STATUSES = new Set([
  "queued",
  "creating",
  "uploading",
  "verifying",
]);

const summarizeResourceUploadProgress = (
  items: ResourceUploadProgress[]
): { inFlight: number; completed: number; failed: number } => {
  let inFlight = 0;
  let completed = 0;
  let failed = 0;
  for (const item of items) {
    const status = String(item.status || "").toLowerCase();
    if (status === "completed") {
      completed += 1;
    } else if (status === "failed") {
      failed += 1;
    } else if (RESOURCE_UPLOAD_IN_FLIGHT_STATUSES.has(status)) {
      inFlight += 1;
    }
  }
  return { inFlight, completed, failed };
};

const LazyUploadViewerSheet = lazyNamed(
  loadUploadViewerSheetModule,
  "UploadViewerSheet"
);
const LazyAdminConsole = lazyNamed(
  loadAdminConsoleModule,
  "AdminConsole"
);
const LazyAppSettingsDialog = lazyNamed(
  loadAppSettingsDialogModule,
  "AppSettingsDialog"
);
const LazyAuthScreen = lazyNamed(loadAuthScreenModule, "AuthScreen");
const LazyToaster = lazy(async () => {
  const module = await loadSonnerModule();
  return {
    default: module.Toaster as ComponentType<any>,
  };
});
const LazyTrainingDashboard = lazyNamed(
  loadTrainingDashboardModule,
  "TrainingDashboard"
);
const LazyScientificViewerPage = lazyNamed(
  loadScientificViewerPageModule,
  "ScientificViewerPage"
);
const LazyResourceBrowser = lazyNamed(
  loadResourceBrowserModule,
  "ResourceBrowser"
);
const LazyComposerSlashMenu = lazyNamed(
  loadComposerSlashMenuModule,
  "ComposerSlashMenu"
);
const LazyChatRunSteps = lazyNamed(loadChatRunStepsModule, "ChatRunSteps");
const LazyInlineDataQuickPreview = lazyNamed(
  loadInlineDataQuickPreviewModule,
  "InlineDataQuickPreview"
);
const LazyProModeDevTrace = lazyNamed(loadProModeDevTraceModule, "ProModeDevTrace");
const LazyToolResultCardSection = lazyNamed(
  loadToolResultCardsModule,
  "ToolResultCardSection"
);
const LazyResearchDigestCard = lazyNamed(
  loadToolResultCardsModule,
  "ResearchDigestCard"
);

let secondaryPanelPreloadPromise: Promise<unknown[]> | null = null;
let adminPanelPreloadPromise: Promise<unknown> | null = null;

const preloadAdminPanelModule = (): Promise<unknown> => {
  adminPanelPreloadPromise ??= loadAdminConsoleModule().catch((error: unknown) => {
    adminPanelPreloadPromise = null;
    throw error;
  });
  return adminPanelPreloadPromise;
};

const preloadSecondaryPanelModules = ({
  includeAdmin = false,
}: { includeAdmin?: boolean } = {}): Promise<unknown[]> => {
  secondaryPanelPreloadPromise ??= Promise.all([
    loadResourceBrowserModule(),
    loadScientificViewerPageModule(),
    loadTrainingDashboardModule(),
    loadAppSettingsDialogModule(),
    loadUploadViewerSheetModule(),
  ]).catch((error: unknown) => {
    secondaryPanelPreloadPromise = null;
    throw error;
  });
  if (!includeAdmin) {
    return secondaryPanelPreloadPromise;
  }
  return Promise.all([
    secondaryPanelPreloadPromise,
    preloadAdminPanelModule(),
  ]).then(([secondaryModules, adminModule]) => [
    ...secondaryModules,
    adminModule,
  ]);
};

type UiMessage = {
  id: string;
  role: UiRole;
  content: string;
  createdAt: number;
  runId?: string;
  durationSeconds?: number;
  progressEvents?: ProgressEvent[];
  runEvents?: RunEvent[];
  responseMetadata?: Record<string, unknown> | null;
  uploadedFileNames?: string[];
  liveStream?: AsyncIterable<string>;
  runArtifacts?: RunImageArtifact[];
  quickPreviewFileIds?: string[];
  resolvedBisqueResources?: ToolResourceRow[];
};

type HistoryPeriod = "Today" | "Yesterday" | "Last 7 days" | "Older";

type HistoryItem = {
  id: string;
  title: string;
  preview: string;
  period: HistoryPeriod;
  running: boolean;
  messageCount: number;
};

type CollapsedSidebarRailProps = {
  recentItems: HistoryItem[];
  activeConversationId: string | null;
  resourcesActive: boolean;
  onCreateConversation: () => void;
  onOpenResources: () => void;
  onOpenRecent: (conversation: HistoryItem) => void;
};

function CollapsedSidebarRail({
  recentItems,
  activeConversationId,
  resourcesActive,
  onCreateConversation,
  onOpenResources,
  onOpenRecent,
}: CollapsedSidebarRailProps) {
  const [recentsOpen, setRecentsOpen] = useState(false);

  return (
    <nav className="app-collapsed-sidebar-rail" aria-label="Collapsed navigation">
      <SidebarTrigger
        className="app-collapsed-sidebar-toggle"
        aria-label="Expand sidebar"
        title="Expand sidebar"
      />

      <div className="app-collapsed-sidebar-actions" role="list">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="app-collapsed-sidebar-action"
          aria-label="New chat"
          title="New chat"
          onClick={onCreateConversation}
        >
          <PlusIcon data-icon="inline-start" />
        </Button>

        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="app-collapsed-sidebar-action"
          aria-label="Resources"
          title="Resources"
          data-active={resourcesActive ? "true" : undefined}
          onClick={onOpenResources}
        >
          <FolderOpen data-icon="inline-start" />
        </Button>

        <Popover open={recentsOpen} onOpenChange={setRecentsOpen}>
          <PopoverTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="app-collapsed-sidebar-action"
              aria-label="Recents"
              title="Recents"
              data-active={recentsOpen ? "true" : undefined}
            >
              <MessageCircle data-icon="inline-start" />
            </Button>
          </PopoverTrigger>
          <PopoverContent
            side="right"
            align="center"
            sideOffset={8}
            className="app-collapsed-recents-popover"
          >
            <div className="app-collapsed-recents-header">Recents</div>
            <div className="app-collapsed-recents-list">
              {recentItems.length > 0 ? (
                recentItems.map((conversation) => (
                  <button
                    key={conversation.id}
                    type="button"
                    className="app-collapsed-recent-item"
                    data-active={conversation.id === activeConversationId ? "true" : undefined}
                    onClick={() => {
                      onOpenRecent(conversation);
                      setRecentsOpen(false);
                    }}
                  >
                    <span className="truncate">{conversation.title}</span>
                    {conversation.running ? <RunningStatusPill size="compact" /> : null}
                  </button>
                ))
              ) : (
                <div className="app-collapsed-recents-empty">No recent chats yet</div>
              )}
            </div>
          </PopoverContent>
        </Popover>
      </div>
    </nav>
  );
}

type BisqueViewerLink = {
  clientViewUrl: string;
  resourceUri?: string | null;
  imageServiceUrl?: string | null;
  inputUrl?: string;
};

type ConversationState = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  hydrated: boolean;
  historyPreview: string;
  historyMessageCount: number;
  historyRunning: boolean;
  prompt: string;
  messages: UiMessage[];
  pendingFiles: File[];
  uploadedFiles: UploadedFileRecord[];
  stagedUploadFileIds: string[];
  activeSelectionContext: SelectionContext | null;
  failedUploadPreviewIds: Record<string, true>;
  bisqueLinksByFileId: Record<string, BisqueViewerLink>;
  composerWorkflowPreset: ComposerWorkflowPresetState | null;
  selectionImportPending: boolean;
  sending: boolean;
  chatError: string | null;
  streamingMessageId: string | null;
};

const EMPTY_UI_MESSAGES: UiMessage[] = [];
const EMPTY_PROGRESS_EVENTS: ProgressEvent[] = [];
const EMPTY_RUN_EVENTS: RunEvent[] = [];
const EMPTY_RUN_IMAGE_ARTIFACTS: RunImageArtifact[] = [];
const EMPTY_FILES: File[] = [];
const EMPTY_UPLOADED_FILES: UploadedFileRecord[] = [];
const EMPTY_STRING_ARRAY: string[] = [];
const EMPTY_FAILED_UPLOAD_PREVIEW_IDS: Record<string, true> = {};
const EMPTY_BISQUE_LINKS_BY_FILE_ID: Record<string, BisqueViewerLink> = {};

const mergeRunImageArtifacts = (
  primary: RunImageArtifact[],
  secondary: RunImageArtifact[]
): RunImageArtifact[] => {
  const visualPrimary = primary.filter(isHydratableRunArtifactVisual);
  const visualSecondary = secondary.filter(isHydratableRunArtifactVisual);
  if (visualSecondary.length === 0) {
    return visualPrimary;
  }
  const merged = new Map<string, RunImageArtifact>();
  [...visualPrimary, ...visualSecondary].forEach((artifact) => {
    const key = [
      String(artifact.path || "").trim(),
      String(artifact.downloadUrl || artifact.url || "").trim(),
    ]
      .filter(Boolean)
      .join("|");
    if (!key || merged.has(key)) {
      return;
    }
    merged.set(key, artifact);
  });
  return Array.from(merged.values());
};

const collectConversationRunArtifacts = (messages: UiMessage[]): RunImageArtifact[] => {
  const artifacts: RunImageArtifact[] = [];
  messages.forEach((message) => {
    if (message.role !== "assistant" || !Array.isArray(message.runArtifacts)) {
      return;
    }
    artifacts.push(...message.runArtifacts);
  });
  return artifacts.length > 0 ? mergeRunImageArtifacts([], artifacts) : EMPTY_RUN_IMAGE_ARTIFACTS;
};

type ConversationScrollMemory = {
  scrollTop: number;
  wasNearBottom: boolean;
};

const mobileSidebarCloseProps = {
  "data-sidebar-close-mobile": "true",
} as const;

const mobileSidebarKeepOpenProps = {
  "data-sidebar-close-mobile": "false",
} as const;

const CONVERSATION_QUERY_PARAM = "conversation";
const CONVERSATION_PAGE_SIZE = 25;
const RESOURCE_PAGE_SIZE = 50;
const SCROLL_RESTORE_BOTTOM_THRESHOLD_PX = 280;

const parseResourceTagFilter = (value: string): string[] => {
  const seen = new Set<string>();
  const tags: string[] = [];
  value.split(/[\n,]+/).forEach((tag) => {
    const trimmed = tag.trim();
    if (!trimmed || seen.has(trimmed)) {
      return;
    }
    seen.add(trimmed);
    tags.push(trimmed);
  });
  return tags;
};

const captureConversationScrollMemory = (
  scrollElement: HTMLElement
): ConversationScrollMemory => {
  const maxScrollTop = Math.max(scrollElement.scrollHeight - scrollElement.clientHeight, 0);
  const scrollTop = Math.min(Math.max(scrollElement.scrollTop, 0), maxScrollTop);
  return {
    scrollTop,
    wasNearBottom: maxScrollTop - scrollTop <= SCROLL_RESTORE_BOTTOM_THRESHOLD_PX,
  };
};

const readConversationIdFromLocation = (): string | null => {
  if (typeof window === "undefined") {
    return null;
  }
  const value = new URLSearchParams(window.location.search).get(CONVERSATION_QUERY_PARAM);
  return value && value.trim().length > 0 ? value.trim() : null;
};

const buildConversationUrl = (conversationId: string): string => {
  const normalizedConversationId = conversationId.trim();
  if (!normalizedConversationId) {
    return typeof window === "undefined" ? "/" : window.location.origin;
  }
  if (typeof window === "undefined") {
    return `/?${CONVERSATION_QUERY_PARAM}=${encodeURIComponent(normalizedConversationId)}`;
  }
  const nextUrl = new URL(window.location.href);
  nextUrl.searchParams.set(CONVERSATION_QUERY_PARAM, normalizedConversationId);
  nextUrl.searchParams.delete("auth_error");
  return nextUrl.toString();
};

const replaceConversationIdInLocation = (conversationId: string | null): void => {
  if (typeof window === "undefined") {
    return;
  }
  const nextUrl = new URL(window.location.href);
  const normalizedConversationId = String(conversationId || "").trim();
  if (normalizedConversationId) {
    nextUrl.searchParams.set(CONVERSATION_QUERY_PARAM, normalizedConversationId);
  } else {
    nextUrl.searchParams.delete(CONVERSATION_QUERY_PARAM);
  }
  const nextRelativeUrl = `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`;
  const currentRelativeUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (nextRelativeUrl === currentRelativeUrl) {
    return;
  }
  window.history.replaceState({}, "", nextRelativeUrl);
};

const COMPOSER_DRAFTS_STORAGE_KEY = "bisque.frontend.composerDrafts";
const RESOURCE_UPLOAD_PROGRESS_STORAGE_KEY = "bisque.frontend.resourceUploadProgress.v2";
const resourceUploadQueueStore = createResourceUploadQueueStore();

const readComposerDraftsFromStorage = (): Record<string, string> => {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    const raw = window.localStorage.getItem(COMPOSER_DRAFTS_STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed)
        .filter((entry): entry is [string, string] => typeof entry[1] === "string")
        .map(([conversationId, draft]) => [String(conversationId), draft])
    );
  } catch {
    return {};
  }
};

const normalizeResourceUploadProgress = (value: unknown): ResourceUploadProgress | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  const id = String(record.id ?? "").trim();
  const fileName = String(record.fileName ?? "").trim();
  if (!id || !fileName) {
    return null;
  }
  return {
    id,
    fingerprint:
      typeof record.fingerprint === "string" && record.fingerprint.trim()
        ? record.fingerprint.trim()
        : null,
    sessionId:
      typeof record.sessionId === "string" && record.sessionId.trim()
        ? record.sessionId.trim()
        : null,
    fileToken:
      typeof record.fileToken === "string" && record.fileToken.trim()
        ? record.fileToken.trim()
        : null,
    fileName,
    relativePath:
      typeof record.relativePath === "string" && record.relativePath.trim()
        ? record.relativePath.trim()
        : null,
    status: String(record.status ?? "uploading"),
    totalBytes: Math.max(0, Math.floor(Number(record.totalBytes) || 0)),
    bytesVerified: Math.max(0, Math.floor(Number(record.bytesVerified) || 0)),
    error: typeof record.error === "string" && record.error.trim() ? record.error.trim() : null,
  };
};

const readResourceUploadProgressFromStorage = (): ResourceUploadProgress[] => {
  if (typeof window === "undefined") {
    return [];
  }
  try {
    const raw = window.localStorage.getItem(RESOURCE_UPLOAD_PROGRESS_STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .map(normalizeResourceUploadProgress)
      .filter((item): item is ResourceUploadProgress => Boolean(item))
      .slice(0, 12);
  } catch {
    return [];
  }
};

const writeResourceUploadProgressToStorage = (items: ResourceUploadProgress[]): void => {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(RESOURCE_UPLOAD_PROGRESS_STORAGE_KEY, JSON.stringify(items.slice(0, 12)));
  } catch {
    // Local persistence is best effort; backend upload sessions remain authoritative.
  }
};

type ResourceViewerContext = {
  uploadedFiles: UploadedFileRecord[];
  bisqueLinksByFileId: Record<string, BisqueViewerLink>;
};

type PendingConversationDelete = {
  id: string;
  title: string;
};

type PendingConversationRename = {
  id: string;
  title: string;
};

type ConversationHistoryActionsProps = {
  conversationId: string;
  conversationTitle: string;
  deleting: boolean;
  renaming: boolean;
  onCopyLink: (conversationId: string) => Promise<void>;
  onCopyId: (conversationId: string) => Promise<void>;
  onRename: (conversationId: string, conversationTitle: string) => void;
  onDelete: (conversationId: string) => void;
};

type ReuseDecision = "load" | "rerun";

type ReuseCandidateRun = {
  runId: string;
  suggestions: ResourceComputationSuggestion[];
  matchType: "sha256" | "filename";
  toolNames: string[];
  conversationTitle?: string | null;
  conversationUpdatedAt?: string | null;
};

type PendingReusePrompt = {
  suggestions: ResourceComputationSuggestion[];
  candidate: ReuseCandidateRun;
};

type BisqueReferenceSelection = {
  sourceRows: ToolResourceRow[];
  selectedRows: ToolResourceRow[];
  intent: "preview" | "selection";
};

type BisqueImportedSelection = {
  uploadedFiles: UploadedFileRecord[];
  bisqueLinksByFileId: Record<string, BisqueViewerLink>;
};

const ConversationHistoryActions = ({
  conversationId,
  conversationTitle,
  deleting,
  renaming,
  onCopyLink,
  onCopyId,
  onRename,
  onDelete,
}: ConversationHistoryActionsProps) => {
  const { isMobile } = useSidebar();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <SidebarMenuAction asChild showOnHover {...mobileSidebarKeepOpenProps}>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={`Conversation actions for ${conversationTitle}`}
            disabled={deleting}
            className="app-history-action-button size-7 rounded-md border border-transparent bg-transparent p-0 text-muted-foreground shadow-none hover:bg-sidebar-accent hover:text-sidebar-accent-foreground data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
          >
            <MoreHorizontal />
            <span className="sr-only">Conversation actions</span>
          </Button>
        </SidebarMenuAction>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        className="w-52 rounded-lg"
        side={isMobile ? "bottom" : "right"}
        align={isMobile ? "end" : "start"}
        sideOffset={8}
      >
        <DropdownMenuItem
          disabled={deleting || renaming}
          onClick={() => {
            if (deleting || renaming) {
              return;
            }
            onRename(conversationId, conversationTitle);
          }}
        >
          <Pencil className="text-muted-foreground" />
          <span>Rename chat</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => void onCopyLink(conversationId)}>
          <Link2 className="text-muted-foreground" />
          <span>Copy chat link</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => void onCopyId(conversationId)}>
          <Copy className="text-muted-foreground" />
          <span>Copy chat ID</span>
        </DropdownMenuItem>
        <DropdownMenuItem
          variant="destructive"
          disabled={deleting}
          onClick={() => {
            if (deleting) {
              return;
            }
            onDelete(conversationId);
          }}
        >
          <Trash />
          <span>Delete chat</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

type ConversationRenameEditorProps = {
  conversation: HistoryItem;
  value: string;
  disabled: boolean;
  onTitleChange: (conversationId: string, title: string) => void;
  onSubmit: () => Promise<void>;
  onCancel: () => void;
};

const ConversationRenameEditor = memo(function ConversationRenameEditor({
  conversation,
  value,
  disabled,
  onTitleChange,
  onSubmit,
  onCancel,
}: ConversationRenameEditorProps) {
  return (
    <div className="app-history-rename-shell">
      <Input
        value={value}
        onChange={(event) => {
          onTitleChange(conversation.id, event.target.value);
        }}
        onFocus={(event) => {
          event.currentTarget.select();
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            void onSubmit();
          } else if (event.key === "Escape") {
            event.preventDefault();
            onCancel();
          }
        }}
        autoFocus
        maxLength={120}
        aria-label={`Rename ${conversation.title}`}
        data-testid="conversation-rename-input"
        className="app-history-rename-input"
        disabled={disabled}
      />
      <div className="app-history-rename-actions">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="app-history-rename-button"
          aria-label="Save chat name"
          onClick={() => {
            void onSubmit();
          }}
          disabled={disabled}
        >
          <Check className="size-4" />
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="app-history-rename-button"
          aria-label="Cancel renaming chat"
          onClick={onCancel}
          disabled={disabled}
        >
          <X className="size-4" />
        </Button>
      </div>
    </div>
  );
});

type ConversationHistoryRowProps = {
  conversation: HistoryItem;
  active: boolean;
  deleting: boolean;
  renaming: boolean;
  onOpen: (conversation: HistoryItem) => void;
  onCopyLink: (conversationId: string) => Promise<void>;
  onCopyId: (conversationId: string) => Promise<void>;
  onRename: (conversationId: string, conversationTitle: string) => void;
  onDelete: (conversationId: string) => void;
};

const ConversationHistoryRow = memo(function ConversationHistoryRow({
  conversation,
  active,
  deleting,
  renaming,
  onOpen,
  onCopyLink,
  onCopyId,
  onRename,
  onDelete,
}: ConversationHistoryRowProps) {
  return (
    <SidebarMenuItem className="app-history-item">
      <SidebarMenuButton
        isActive={active}
        className="app-history-button group/history h-auto py-2"
        onClick={() => onOpen(conversation)}
        {...mobileSidebarCloseProps}
      >
        <div className="flex min-w-0 w-full items-center gap-2">
          <span className="truncate">{conversation.title}</span>
          <div className="ml-auto flex items-center gap-1.5">
            {conversation.running ? <RunningStatusPill size="compact" /> : null}
          </div>
        </div>
      </SidebarMenuButton>
      <ConversationHistoryActions
        conversationId={conversation.id}
        conversationTitle={conversation.title}
        deleting={deleting}
        renaming={renaming}
        onCopyLink={onCopyLink}
        onCopyId={onCopyId}
        onRename={onRename}
        onDelete={onDelete}
      />
    </SidebarMenuItem>
  );
});

const scientificFileExtensions = [
  ".tif",
  ".tiff",
  ".ome.tif",
  ".ome.tiff",
  ".czi",
  ".nd2",
  ".lif",
  ".lsm",
  ".svs",
  ".vsi",
  ".dv",
  ".r3d",
  ".nii",
  ".nii.gz",
  ".nrrd",
  ".mha",
  ".mhd",
];

const browserPreviewExtensions = new Set([
  "png",
  "jpg",
  "jpeg",
  "gif",
  "bmp",
  "webp",
  "avif",
  "svg",
]);

const NEW_CHAT_SHORTCUT_KEY = "k";
const RESOURCES_SHORTCUT_KEY = "e";
const TRAINING_SHORTCUT_KEY = "t";
const GO_TO_BISQUE_SHORTCUT_KEY = "o";

const isEditableEventTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof Element)) {
    return false;
  }
  if (target instanceof HTMLElement && target.isContentEditable) {
    return true;
  }
  return Boolean(
    target.closest(
      "input, textarea, select, [role='textbox'], [contenteditable=''], [contenteditable='true']"
    )
  );
};

const isScientificUpload = (name: string): boolean => {
  const lowered = name.toLowerCase();
  return scientificFileExtensions.some((suffix) => lowered.endsWith(suffix));
};

const supportsBrowserPreview = (
  name: string,
  contentType?: string | null
): boolean => {
  if (typeof contentType === "string" && contentType.startsWith("image/")) {
    return true;
  }
  const dot = name.lastIndexOf(".");
  if (dot < 0) {
    return false;
  }
  return browserPreviewExtensions.has(name.slice(dot + 1).toLowerCase());
};

const isImageLikeUploadedFile = (file: UploadedFileRecord): boolean => {
  const contentType = String(file.content_type ?? "").toLowerCase();
  if (contentType.startsWith("image/")) {
    return true;
  }
  const lowered = String(file.original_name ?? "").toLowerCase();
  if (scientificFileExtensions.some((suffix) => lowered.endsWith(suffix))) {
    return true;
  }
  const imageLikeExtensions = [
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".nii",
    ".nii.gz",
    ".nrrd",
    ".mha",
    ".mhd",
  ];
  return imageLikeExtensions.some((suffix) => lowered.endsWith(suffix));
};

type PromptWorkflowIntent = {
  asksForSegmentation: boolean;
  asksForDepth: boolean;
  asksForDetection: boolean;
  isMultiToolWorkflow: boolean;
};

const inferPromptWorkflowIntent = (promptText: string): PromptWorkflowIntent => {
  const normalized = String(promptText || "").toLowerCase();
  const asksForSegmentation =
    /\b(segment|segmentation|mask|masks)\b/.test(normalized);
  const asksForDepth = /\b(depth|depth map|depth estimation|depthpro|monocular depth)\b/.test(
    normalized
  );
  const asksForDetection =
    /\b(yolo|detect|detection|object detection|bbox|bounding boxes?)\b/.test(normalized);
  const referencesPipeline =
    /\b(and then|then|after|follow(?:ed)? by|next|pipeline|chain|use .* output|using .* output|take .* output|feed .* into)\b/.test(
      normalized
    );
  const requestedDomains = Number(asksForSegmentation) + Number(asksForDepth) + Number(asksForDetection);
  const isMultiToolWorkflow =
    requestedDomains > 1 ||
    (asksForSegmentation && referencesPipeline && (asksForDepth || asksForDetection));

  return {
    asksForSegmentation,
    asksForDepth,
    asksForDetection,
    isMultiToolWorkflow,
  };
};

const PRO_MODE_COMPOSER_WORKFLOW_PRESET: ComposerWorkflowPresetState = {
  id: "pro_mode",
  label: "Pro Mode",
  prompt: "",
  selectedToolNames: [],
  workflowHint: {
    id: "pro_mode",
    source: "slash_menu",
  },
  requiresAttachedFiles: false,
  opensResourcePickerOnSelect: false,
  clearsAfterResourcePick: false,
  persistsAcrossTurns: true,
};

const toComposerWorkflowPresetState = (
  workflow: ComposerWorkflowPresetState
): ComposerWorkflowPresetState => ({
  id: workflow.id,
  label: workflow.label,
  prompt: workflow.prompt,
  selectedToolNames: [...workflow.selectedToolNames],
  workflowHint: workflow.workflowHint ? { ...workflow.workflowHint } : null,
  requiresAttachedFiles: workflow.requiresAttachedFiles,
  opensResourcePickerOnSelect: workflow.opensResourcePickerOnSelect,
  clearsAfterResourcePick: workflow.clearsAfterResourcePick,
  persistsAcrossTurns: workflow.persistsAcrossTurns ?? false,
});

const coerceComposerWorkflowPresetState = (
  value: unknown
): ComposerWorkflowPresetState | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  const id = String(record.id ?? "").trim() as ComposerWorkflowId;
  const label = String(record.label ?? "").trim();
  if (!id || !label) {
    return null;
  }
  const selectedToolNames = Array.isArray(record.selectedToolNames)
    ? record.selectedToolNames.map((item) => String(item || "").trim()).filter(Boolean)
    : [];
  const hintRecord =
    record.workflowHint &&
    typeof record.workflowHint === "object" &&
    !Array.isArray(record.workflowHint)
      ? (record.workflowHint as Record<string, unknown>)
      : null;
  const hintId = String(hintRecord?.id ?? "").trim();
  return {
    id,
    label,
    prompt: String(record.prompt ?? ""),
    selectedToolNames,
    workflowHint: hintId
      ? ({
          id: hintId as ComposerWorkflowId,
          source: "slash_menu",
        } as NonNullable<ComposerWorkflowPresetState["workflowHint"]>)
      : null,
    requiresAttachedFiles: Boolean(record.requiresAttachedFiles),
    opensResourcePickerOnSelect: Boolean(record.opensResourcePickerOnSelect),
    clearsAfterResourcePick: Boolean(record.clearsAfterResourcePick),
    persistsAcrossTurns: Boolean(record.persistsAcrossTurns),
  };
};

const inferReuseToolNames = (promptText: string): string[] => {
  const intent = inferPromptWorkflowIntent(promptText);
  const selected: string[] = [];
  if (intent.asksForDetection) {
    selected.push("yolo_detect");
  }
  if (intent.asksForDepth) {
    selected.push("estimate_depth_pro");
  }
  return selected;
};

const normalizeSlashWorkflowQuery = (value: string): string => {
  const prompt = String(value || "");
  if (!prompt.startsWith("/")) {
    return "";
  }
  return prompt.slice(1).trim();
};

const reuseToolLabel = (toolName: string): string => {
  if (toolName === "yolo_detect") {
    return "YOLO detection";
  }
  if (toolName === "estimate_depth_pro") {
    return "Depth estimation";
  }
  return toolName;
};

const autoLoadReuseToolNames = new Set<string>([
  "yolo_detect",
  "estimate_depth_pro",
]);

const promptExplicitlyRequestsReuseLoad = (promptText: string): boolean => {
  const lowered = String(promptText || "").trim().toLowerCase();
  if (!lowered) {
    return false;
  }
  return (
    /\b(load|reuse|use|open|show|continue with|work from)\b.{0,32}\b(previous|prior|cached|existing)\b/.test(
      lowered
    ) ||
    /\b(load|reuse|use)\b.{0,24}\b(results?|analysis|run|output|outputs)\b/.test(lowered)
  );
};

const promptRequestsFreshReuseComputation = (promptText: string): boolean => {
  const lowered = String(promptText || "").trim().toLowerCase();
  if (!lowered) {
    return false;
  }
  return /\b(run again|rerun|re-run|recompute|fresh|from scratch|new analysis|new run)\b/.test(
    lowered
  );
};

const parseIsoTimestamp = (value: string | null | undefined): number => {
  const parsed = Date.parse(String(value ?? "").trim());
  return Number.isFinite(parsed) ? parsed : 0;
};

const formatReuseTimestamp = (value: string | null | undefined): string | null => {
  const parsed = parseIsoTimestamp(value);
  if (!parsed) {
    return null;
  }
  return new Date(parsed).toLocaleString();
};

const reuseSimilarityStopWords = new Set([
  "the",
  "and",
  "with",
  "this",
  "that",
  "from",
  "into",
  "your",
  "their",
  "then",
  "than",
  "have",
  "make",
  "some",
  "more",
  "same",
  "image",
  "file",
  "chat",
  "please",
  "using",
  "used",
  "show",
  "tell",
  "give",
  "also",
  "what",
  "does",
  "mean",
  "about",
  "below",
  "above",
  "around",
  "again",
  "run",
]);

const tokenizeReusePrompt = (value: string): string[] =>
  Array.from(
    new Set(
      String(value || "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, " ")
        .split(/\s+/)
        .map((token) => token.trim())
        .filter(
          (token) => token.length >= 3 && !reuseSimilarityStopWords.has(token)
        )
    )
  );

const reuseGoalSimilarityScore = (
  promptText: string,
  suggestion: ResourceComputationSuggestion
): number => {
  const promptTokens = tokenizeReusePrompt(promptText);
  if (promptTokens.length === 0) {
    return 0;
  }
  const candidateTokens = new Set(
    tokenizeReusePrompt(
      `${String(suggestion.run_goal || "").trim()} ${String(
        suggestion.conversation_title || ""
      ).trim()}`
    )
  );
  if (candidateTokens.size === 0) {
    return 0;
  }
  const overlap = promptTokens.filter((token) => candidateTokens.has(token)).length;
  return overlap / promptTokens.length;
};

const selectReuseCandidateRun = (
  suggestions: ResourceComputationSuggestion[],
  promptText: string
): ReuseCandidateRun | null => {
  if (suggestions.length === 0) {
    return null;
  }
  const grouped = new Map<
    string,
    {
      suggestions: ResourceComputationSuggestion[];
      toolNames: Set<string>;
      hasShaMatch: boolean;
      latestUpdatedAt: number;
      conversationTitle?: string | null;
      conversationUpdatedAt?: string | null;
      bestPromptSimilarity: number;
    }
  >();

  suggestions.forEach((suggestion) => {
    const runId = String(suggestion.run_id || "").trim();
    if (!runId) {
      return;
    }
    const existing = grouped.get(runId) ?? {
      suggestions: [],
      toolNames: new Set<string>(),
      hasShaMatch: false,
      latestUpdatedAt: 0,
      conversationTitle: null,
      conversationUpdatedAt: null,
      bestPromptSimilarity: 0,
    };
    existing.suggestions.push(suggestion);
    existing.toolNames.add(String(suggestion.tool_name || "").trim());
    if (suggestion.match_type === "sha256") {
      existing.hasShaMatch = true;
    }
    existing.latestUpdatedAt = Math.max(
      existing.latestUpdatedAt,
      parseIsoTimestamp(suggestion.run_updated_at)
    );
    existing.bestPromptSimilarity = Math.max(
      existing.bestPromptSimilarity,
      reuseGoalSimilarityScore(promptText, suggestion)
    );
    if (!existing.conversationTitle && suggestion.conversation_title) {
      existing.conversationTitle = suggestion.conversation_title;
    }
    if (!existing.conversationUpdatedAt && suggestion.conversation_updated_at) {
      existing.conversationUpdatedAt = suggestion.conversation_updated_at;
    }
    grouped.set(runId, existing);
  });

  const ranked = Array.from(grouped.entries()).sort((left, right) => {
    const leftPayload = left[1];
    const rightPayload = right[1];
    if (Number(rightPayload.hasShaMatch) !== Number(leftPayload.hasShaMatch)) {
      return Number(rightPayload.hasShaMatch) - Number(leftPayload.hasShaMatch);
    }
    if (rightPayload.bestPromptSimilarity !== leftPayload.bestPromptSimilarity) {
      return rightPayload.bestPromptSimilarity - leftPayload.bestPromptSimilarity;
    }
    if (rightPayload.suggestions.length !== leftPayload.suggestions.length) {
      return rightPayload.suggestions.length - leftPayload.suggestions.length;
    }
    return rightPayload.latestUpdatedAt - leftPayload.latestUpdatedAt;
  });
  if (ranked.length === 0) {
    return null;
  }
  const [runId, payload] = ranked[0];
  return {
    runId,
    suggestions: payload.suggestions,
    matchType: payload.hasShaMatch ? "sha256" : "filename",
    toolNames: Array.from(payload.toolNames).filter((toolName) => toolName.length > 0),
    conversationTitle: payload.conversationTitle,
    conversationUpdatedAt: payload.conversationUpdatedAt,
  };
};

const shouldAutoLoadReuseCandidate = (
  promptText: string,
  candidate: ReuseCandidateRun
): boolean => {
  if (candidate.matchType !== "sha256") {
    return false;
  }
  if (promptRequestsFreshReuseComputation(promptText)) {
    return false;
  }
  if (candidate.toolNames.length === 0) {
    return false;
  }
  if (!promptExplicitlyRequestsReuseLoad(promptText)) {
    return false;
  }
  return candidate.toolNames.every((toolName) => autoLoadReuseToolNames.has(toolName));
};

const makeId = (): string =>
  typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

const createConversationState = (): ConversationState => {
  const now = Date.now();
  const id = makeId();
  return {
    id,
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    hydrated: true,
    historyPreview: "",
    historyMessageCount: 0,
    historyRunning: false,
    prompt: "",
    messages: [],
    pendingFiles: [],
    uploadedFiles: [],
    stagedUploadFileIds: [],
    activeSelectionContext: null,
    failedUploadPreviewIds: {},
    bisqueLinksByFileId: {},
    composerWorkflowPreset: toComposerWorkflowPresetState(
      PRO_MODE_COMPOSER_WORKFLOW_PRESET
    ),
    selectionImportPending: false,
    sending: false,
    chatError: null,
    streamingMessageId: null,
  };
};

const toMillis = (value: unknown, fallback: number): number => {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return Math.floor(value);
  }
  if (typeof value === "string" && /^\d+$/.test(value.trim())) {
    return Math.max(0, Number.parseInt(value, 10));
  }
  return fallback;
};

const toUiRole = (value: unknown): UiRole =>
  String(value || "").toLowerCase() === "user" ? "user" : "assistant";

const toUploadedFileRecords = (value: unknown): UploadedFileRecord[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((row): row is Record<string, unknown> => Boolean(row && typeof row === "object"))
    .map((row) => ({
      file_id: String(row.file_id || ""),
      original_name: String(row.original_name || "upload.bin"),
      content_type: row.content_type ? String(row.content_type) : null,
      size_bytes: Math.max(0, Number(row.size_bytes) || 0),
      sha256: String(row.sha256 || ""),
      created_at: String(row.created_at || new Date().toISOString()),
    }))
    .filter((row) => row.file_id.length > 0);
};

const toFileIdList = (value: unknown): string[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const seen = new Set<string>();
  const ordered: string[] = [];
  value.forEach((entry) => {
    const fileId = String(entry || "").trim();
    if (!fileId || seen.has(fileId)) {
      return;
    }
    seen.add(fileId);
    ordered.push(fileId);
  });
  return ordered;
};

const toArtifactHandleMap = (value: unknown): Record<string, string[]> => {
  if (!value || typeof value !== "object") {
    return {};
  }
  const record = value as Record<string, unknown>;
  const normalized: Record<string, string[]> = {};
  Object.entries(record).forEach(([key, rawValues]) => {
    const handleKey = String(key || "").trim();
    if (!handleKey) {
      return;
    }
    const values = Array.isArray(rawValues) ? rawValues : [rawValues];
    const cleaned = values
      .map((entry) => String(entry || "").trim())
      .filter((entry) => entry.length > 0);
    if (cleaned.length > 0) {
      normalized[handleKey] = Array.from(new Set(cleaned));
    }
  });
  return normalized;
};

const toSelectionContext = (value: unknown): SelectionContext | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const normalized: SelectionContext = {
    context_id:
      typeof record.context_id === "string" && record.context_id.trim()
        ? record.context_id.trim()
        : null,
    source:
      typeof record.source === "string" && record.source.trim() ? record.source.trim() : null,
    focused_file_ids: toFileIdList(record.focused_file_ids),
    resource_uris: Array.isArray(record.resource_uris)
      ? record.resource_uris
          .map((entry) => String(entry || "").trim())
          .filter((entry) => entry.length > 0)
      : [],
    dataset_uris: Array.isArray(record.dataset_uris)
      ? record.dataset_uris
          .map((entry) => String(entry || "").trim())
          .filter((entry) => entry.length > 0)
      : [],
    artifact_handles: toArtifactHandleMap(record.artifact_handles),
    originating_message_id:
      typeof record.originating_message_id === "string" && record.originating_message_id.trim()
        ? record.originating_message_id.trim()
        : null,
    originating_user_text:
      typeof record.originating_user_text === "string" && record.originating_user_text.trim()
        ? record.originating_user_text.trim()
        : null,
    suggested_domain:
      typeof record.suggested_domain === "string" && record.suggested_domain.trim()
        ? record.suggested_domain.trim()
        : null,
    suggested_tool_names: Array.isArray(record.suggested_tool_names)
      ? record.suggested_tool_names
          .map((entry) => String(entry || "").trim())
          .filter((entry) => entry.length > 0)
      : [],
  };
  if (
    !normalized.context_id &&
    !normalized.source &&
    (normalized.focused_file_ids?.length ?? 0) === 0 &&
    (normalized.resource_uris?.length ?? 0) === 0 &&
    (normalized.dataset_uris?.length ?? 0) === 0 &&
    Object.keys(normalized.artifact_handles ?? {}).length === 0 &&
    !normalized.originating_message_id &&
    !normalized.originating_user_text &&
    !normalized.suggested_domain &&
    (normalized.suggested_tool_names?.length ?? 0) === 0
  ) {
    return null;
  }
  return normalized;
};

const toRunArtifacts = (value: unknown): RunImageArtifact[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((row): row is Record<string, unknown> => Boolean(row && typeof row === "object"))
    .map((row) => ({
      path: String(row.path || ""),
      url: String(row.url || ""),
      title: String(row.title || "Artifact"),
      sourceName: String(row.sourceName || ""),
      sourcePath:
        String((row.sourcePath ?? row.source_path ?? "") || "").trim() || undefined,
      previewable: Boolean(row.previewable),
      downloadUrl: String((row.downloadUrl ?? row.download_url ?? row.url) || ""),
      linkedFileId:
        typeof row.linkedFileId === "string"
          ? row.linkedFileId
          : typeof row.linked_file_id === "string"
            ? row.linked_file_id
            : null,
    }))
    .filter(
      (artifact) =>
        artifact.path.length > 0 &&
        artifact.url.length > 0 &&
        isHydratableRunArtifactVisual(artifact)
    );
};

const toToolResourceRows = (value: unknown): ToolResourceRow[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((entry) => toRecord(entry))
    .filter((entry): entry is Record<string, unknown> => entry !== null)
    .flatMap((entry) => {
      const name = String(entry.name ?? "").trim();
      const resourceUri = String(entry.resourceUri ?? entry.resource_uri ?? "").trim();
      const clientViewUrl =
        String(entry.clientViewUrl ?? entry.client_view_url ?? "").trim() ||
        toBisqueClientViewUrl(resourceUri) ||
        "";
      const imageServiceUrl = String(
        entry.imageServiceUrl ?? entry.image_service_url ?? ""
      ).trim();
      const uri = String(entry.uri ?? clientViewUrl ?? resourceUri ?? "").trim();
      if (!name && !resourceUri && !clientViewUrl && !uri) {
        return [];
      }
      return [
        {
          name: name || "resource",
          owner: String(entry.owner ?? "").trim() || undefined,
          created: String(entry.created ?? "").trim() || undefined,
          resourceType: String(entry.resourceType ?? entry.resource_type ?? "").trim() || undefined,
          uri: uri || undefined,
          resourceUri: resourceUri || undefined,
          clientViewUrl: clientViewUrl || undefined,
          imageServiceUrl: imageServiceUrl || undefined,
        } satisfies ToolResourceRow,
      ];
    });
};

const toProgressEvents = (value: unknown): ProgressEvent[] =>
  Array.isArray(value)
    ? value.filter((row): row is ProgressEvent => Boolean(row && typeof row === "object"))
    : [];

const toRunEvents = (value: unknown): RunEvent[] =>
  Array.isArray(value)
    ? value.filter((row): row is RunEvent => Boolean(row && typeof row === "object"))
    : [];

const toBisqueLinks = (value: unknown): Record<string, BisqueViewerLink> => {
  if (!value || typeof value !== "object") {
    return {};
  }
  const entries = Object.entries(value as Record<string, unknown>);
  const output: Record<string, BisqueViewerLink> = {};
  entries.forEach(([fileId, payload]) => {
    if (!payload || typeof payload !== "object") {
      return;
    }
    const row = payload as Record<string, unknown>;
    const clientViewUrl = String(row.clientViewUrl || "").trim();
    if (!clientViewUrl) {
      return;
    }
    output[fileId] = {
      clientViewUrl,
      resourceUri: row.resourceUri ? String(row.resourceUri) : null,
      imageServiceUrl: row.imageServiceUrl ? String(row.imageServiceUrl) : null,
      inputUrl: row.inputUrl ? String(row.inputUrl) : undefined,
    };
  });
  return output;
};

const toFailedPreviewIds = (value: unknown): Record<string, true> => {
  if (!value || typeof value !== "object") {
    return {};
  }
  const output: Record<string, true> = {};
  Object.entries(value as Record<string, unknown>).forEach(([fileId, flag]) => {
    if (flag) {
      output[fileId] = true;
    }
  });
  return output;
};

const toUiMessages = (value: unknown, fallbackTime: number): UiMessage[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((row): row is Record<string, unknown> => Boolean(row && typeof row === "object"))
    .map((row, index) => {
      const role = toUiRole(row.role);
      const rawContent = String(row.content || "");
      return {
        id: String(row.id || `${fallbackTime}-${index}`),
        role,
        content: rawContent,
        createdAt: toMillis(row.createdAt, fallbackTime),
        runId: row.runId ? String(row.runId) : undefined,
        durationSeconds: toNumber(row.durationSeconds ?? row.duration_seconds) ?? undefined,
        progressEvents: toProgressEvents(row.progressEvents),
        runEvents: toRunEvents(row.runEvents),
        responseMetadata: toRecord(row.responseMetadata),
        uploadedFileNames: Array.isArray(row.uploadedFileNames)
          ? row.uploadedFileNames.map((item) => String(item))
          : undefined,
        runArtifacts: toRunArtifacts(row.runArtifacts),
        quickPreviewFileIds: Array.isArray(row.quickPreviewFileIds)
          ? row.quickPreviewFileIds.map((item) => String(item)).filter(Boolean)
          : undefined,
        resolvedBisqueResources: toToolResourceRows(row.resolvedBisqueResources),
      };
    })
    .filter((message) => !shouldDropHydratedLegacy404Message(message));
};

const removeMessageWithPairedResponse = (
  messages: UiMessage[],
  messageId: string
): UiMessage[] => {
  const targetIndex = messages.findIndex((item) => item.id === messageId);
  if (targetIndex < 0) {
    return messages;
  }

  const target = messages[targetIndex];
  const idsToRemove = new Set<string>([target.id]);
  if (target.role === "user") {
    for (let index = targetIndex + 1; index < messages.length; index += 1) {
      const candidate = messages[index];
      if (candidate.role !== "assistant") {
        break;
      }
      idsToRemove.add(candidate.id);
    }
  }

  return messages.filter((item) => !idsToRemove.has(item.id));
};

const conversationFromRecord = (record: ConversationRecord): ConversationState => {
  const now = Date.now();
  const createdAt = toMillis(record.created_at_ms, now);
  const updatedAt = toMillis(record.updated_at_ms, createdAt);
  const state = (record.state || {}) as Record<string, unknown>;
  const hydrated = Object.keys(state).length > 0;
  const conversationId = String(record.conversation_id || makeId());
  const messages = hydrated ? toUiMessages(state.messages, updatedAt) : [];
  const titleSeed = hydrated
    ? [...messages].reverse().find((message) => message.role === "user")?.content ??
      messages[messages.length - 1]?.content ??
      String(record.preview || "")
    : String(record.preview || "");
  const uploadedFiles = hydrated ? toUploadedFileRecords(state.uploadedFiles) : [];
  const uploadedFileIdSet = new Set(uploadedFiles.map((file) => file.file_id));
  const stagedUploadFileIds = hydrated
    ? toFileIdList(state.stagedUploadFileIds).filter((fileId) => uploadedFileIdSet.has(fileId))
    : [];
  return {
    id: conversationId,
    title: resolveConversationTitle(record.title, titleSeed),
    createdAt,
    updatedAt,
    hydrated,
    historyPreview: String(record.preview || "").trim(),
    historyMessageCount:
      typeof record.message_count === "number" && Number.isFinite(record.message_count)
        ? Math.max(0, Math.floor(record.message_count))
        : 0,
    historyRunning: Boolean(record.running),
    prompt: hydrated ? String(state.prompt || "") : "",
    messages,
    pendingFiles: [],
    uploadedFiles,
    stagedUploadFileIds,
    activeSelectionContext: hydrated ? toSelectionContext(state.activeSelectionContext) : null,
    failedUploadPreviewIds: hydrated ? toFailedPreviewIds(state.failedUploadPreviewIds) : {},
    bisqueLinksByFileId: hydrated ? toBisqueLinks(state.bisqueLinksByFileId) : {},
    composerWorkflowPreset: hydrated
      ? coerceComposerWorkflowPresetState(state.composerWorkflowPreset)
      : null,
    selectionImportPending: false,
    sending: hydrated ? Boolean(state.sending) : false,
    chatError:
      hydrated &&
      typeof state.chatError === "string" &&
      state.chatError.trim() &&
      !isLegacyRunLookup404Content(state.chatError)
        ? state.chatError
        : null,
    streamingMessageId:
      hydrated &&
      typeof state.streamingMessageId === "string" &&
      state.streamingMessageId.trim()
        ? state.streamingMessageId
        : null,
  };
};

const mergeConversationPage = (
  existing: ConversationState[],
  incoming: ConversationState[]
): ConversationState[] => {
  const existingById = new Map(existing.map((conversation) => [conversation.id, conversation] as const));
  const merged = incoming.map((candidate) => {
    const current = existingById.get(candidate.id);
    if (!current) {
      return candidate;
    }
    if (current.hydrated && !candidate.hydrated) {
      return {
        ...current,
        title: candidate.title,
        createdAt: candidate.createdAt,
        updatedAt: candidate.updatedAt,
        historyPreview: candidate.historyPreview,
        historyMessageCount: candidate.historyMessageCount,
        historyRunning: candidate.historyRunning,
      };
    }
    if (!current.hydrated && candidate.hydrated) {
      return candidate;
    }
    return current.updatedAt >= candidate.updatedAt ? current : candidate;
  });
  const incomingIds = new Set(incoming.map((conversation) => conversation.id));
  const optimisticOnly = existing.filter((conversation) => !incomingIds.has(conversation.id));
  return [...optimisticOnly, ...merged].sort((a, b) => b.updatedAt - a.updatedAt);
};

const conversationToRecord = (conversation: ConversationState): ConversationRecord => {
  const previewSource = conversation.hydrated
    ? [...conversation.messages].reverse().find((message) => message.role === "user")?.content ??
      conversation.messages[conversation.messages.length - 1]?.content ??
      ""
    : conversation.historyPreview;
  return {
    conversation_id: conversation.id,
    title: resolveConversationTitle(conversation.title, previewSource),
    created_at_ms: conversation.createdAt,
    updated_at_ms: conversation.updatedAt,
    preview: conversation.hydrated ? summarizePrompt(previewSource, 160) : conversation.historyPreview,
    message_count: conversation.hydrated
      ? conversation.messages.length
      : conversation.historyMessageCount,
    preferred_panel: "chat",
    running: conversation.hydrated ? conversation.sending : conversation.historyRunning,
    state: {
      prompt: "",
      messages: conversation.messages.map((message) => ({
        id: message.id,
        role: message.role,
        content: message.content,
        createdAt: message.createdAt,
        runId: message.runId,
        durationSeconds: message.durationSeconds,
        progressEvents: message.progressEvents ?? [],
        runEvents: message.runEvents ?? [],
        responseMetadata: message.responseMetadata ?? null,
        uploadedFileNames: message.uploadedFileNames ?? [],
        runArtifacts: message.runArtifacts ?? [],
        quickPreviewFileIds: message.quickPreviewFileIds ?? [],
        resolvedBisqueResources: message.resolvedBisqueResources ?? [],
      })),
      uploadedFiles: conversation.uploadedFiles,
      stagedUploadFileIds: conversation.stagedUploadFileIds,
      activeSelectionContext: conversation.activeSelectionContext,
      failedUploadPreviewIds: conversation.failedUploadPreviewIds,
      bisqueLinksByFileId: conversation.bisqueLinksByFileId,
      composerWorkflowPreset: conversation.composerWorkflowPreset,
      selectionImportPending: false,
      sending: Boolean(conversation.sending),
      chatError: conversation.chatError,
      streamingMessageId: conversation.streamingMessageId,
    },
  };
};

type StreamController = {
  iterable: AsyncIterable<string>;
  push: (value: string) => void;
  close: () => void;
  fail: (reason?: unknown) => void;
};

const createStreamController = (): StreamController => {
  const chunks: string[] = [];
  const waiters = new Set<() => void>();
  let closed = false;
  let failureReason: unknown = null;

  const notifyWaiters = (): void => {
    if (waiters.size === 0) {
      return;
    }
    Array.from(waiters).forEach((waiter) => waiter());
  };

  const safeClose = (): void => {
    if (closed) {
      return;
    }
    closed = true;
    notifyWaiters();
  };

  const safeError = (reason?: unknown): void => {
    if (closed) {
      return;
    }
    closed = true;
    failureReason = reason ?? new Error("Text stream failed.");
    notifyWaiters();
  };

  return {
    iterable: {
      async *[Symbol.asyncIterator]() {
        let index = 0;
        let pendingWaiter: (() => void) | null = null;
        try {
          while (true) {
            while (index < chunks.length) {
              const value = chunks[index];
              index += 1;
              if (typeof value === "string") {
                yield value;
              }
            }
            if (failureReason) {
              throw failureReason;
            }
            if (closed) {
              break;
            }
            await new Promise<void>((resolve) => {
              const waiter = () => {
                waiters.delete(waiter);
                if (pendingWaiter === waiter) {
                  pendingWaiter = null;
                }
                resolve();
              };
              pendingWaiter = waiter;
              waiters.add(waiter);
            });
          }
        } finally {
          if (pendingWaiter) {
            waiters.delete(pendingWaiter);
          }
        }
      },
    },
    push: (value: string) => {
      if (closed || !value) {
        return;
      }
      chunks.push(value);
      notifyWaiters();
    },
    close: safeClose,
    fail: safeError,
  };
};

const uniqueByFileId = (rows: UploadedFileRecord[]): UploadedFileRecord[] => {
  const mapped = new Map<string, UploadedFileRecord>();
  rows.forEach((row) => mapped.set(row.file_id, row));
  return Array.from(mapped.values());
};

const uniqueFileIds = (rows: string[]): string[] => {
  const seen = new Set<string>();
  const ordered: string[] = [];
  rows.forEach((fileId) => {
    const normalized = String(fileId || "").trim();
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    ordered.push(normalized);
  });
  return ordered;
};

function ChatAutoScroll({
  conversationId,
  conversationHydrated,
  scrollRequestKey,
  scrollMemoryRef,
  scrollElementRef,
  scrollWriteBlockRef,
}: {
  conversationId: string | null;
  conversationHydrated: boolean;
  scrollRequestKey: number;
  scrollMemoryRef: MutableRefObject<Record<string, ConversationScrollMemory>>;
  scrollElementRef: MutableRefObject<HTMLElement | null>;
  scrollWriteBlockRef: MutableRefObject<string | null>;
}) {
  const { scrollRef, scrollToBottom, stopScroll } = useStickToBottomContext();
  const restoredConversationIdRef = useRef<string | null>(null);
  const liveConversationIdRef = useRef<string | null>(conversationId);
  const previousScrollRequestKeyRef = useRef(scrollRequestKey);

  const rememberScrollPosition = useCallback(
    (targetConversationId: string | null) => {
      const scrollElement = scrollRef.current;
      if (!targetConversationId || !scrollElement) {
        return;
      }
      scrollMemoryRef.current[targetConversationId] = captureConversationScrollMemory(scrollElement);
    },
    [scrollMemoryRef, scrollRef]
  );

  useLayoutEffect(() => {
    liveConversationIdRef.current = conversationId;
  }, [conversationId]);

  useLayoutEffect(() => {
    const scrollElement = scrollRef.current;
    scrollElementRef.current = scrollElement;
    return () => {
      if (scrollElementRef.current === scrollElement) {
        scrollElementRef.current = null;
      }
    };
  }, [scrollElementRef, scrollRef]);

  useLayoutEffect(() => {
    if (!conversationId) {
      restoredConversationIdRef.current = null;
      return;
    }
    if (!conversationHydrated || restoredConversationIdRef.current === conversationId) {
      return;
    }
    restoredConversationIdRef.current = conversationId;
    let rafIdOne = 0;
    let rafIdTwo = 0;
    rafIdOne = requestAnimationFrame(() => {
      rafIdTwo = requestAnimationFrame(() => {
        const remembered = scrollMemoryRef.current[conversationId];
        if (remembered && !remembered.wasNearBottom) {
          const scrollElement = scrollRef.current;
          if (!scrollElement) {
            return;
          }
          stopScroll();
          const maxScrollTop = Math.max(scrollElement.scrollHeight - scrollElement.clientHeight, 0);
          scrollElement.scrollTop = Math.min(remembered.scrollTop, maxScrollTop);
          rememberScrollPosition(conversationId);
          if (scrollWriteBlockRef.current === conversationId) {
            scrollWriteBlockRef.current = null;
          }
          return;
        }
        scrollToBottom({ animation: "instant", ignoreEscapes: true });
        if (scrollWriteBlockRef.current === conversationId) {
          scrollWriteBlockRef.current = null;
        }
      });
    });
    return () => {
      if (rafIdOne) {
        cancelAnimationFrame(rafIdOne);
      }
      if (rafIdTwo) {
        cancelAnimationFrame(rafIdTwo);
      }
    };
  }, [
    conversationHydrated,
    conversationId,
    rememberScrollPosition,
    scrollMemoryRef,
    scrollRef,
    scrollToBottom,
    scrollWriteBlockRef,
    stopScroll,
  ]);

  useEffect(() => {
    if (!conversationId || !conversationHydrated) {
      return;
    }
    const scrollElement = scrollRef.current;
    if (!scrollElement) {
      return;
    }
    const handleScroll = () => {
      if (
        liveConversationIdRef.current !== conversationId ||
        scrollWriteBlockRef.current === conversationId
      ) {
        return;
      }
      rememberScrollPosition(conversationId);
    };
    scrollElement.addEventListener("scroll", handleScroll, { passive: true });
    return () => {
      scrollElement.removeEventListener("scroll", handleScroll);
    };
  }, [conversationHydrated, conversationId, rememberScrollPosition, scrollRef, scrollWriteBlockRef]);

  useEffect(() => {
    if (!conversationId || scrollRequestKey === previousScrollRequestKeyRef.current) {
      return;
    }
    previousScrollRequestKeyRef.current = scrollRequestKey;
    scrollToBottom({ animation: "smooth", ignoreEscapes: true });
  }, [conversationId, scrollRequestKey, scrollToBottom]);

  return null;
}

type ConversationTranscriptActions = {
  onStopConversation: () => void;
  onStreamingRenderComplete: (messageId: string) => void;
  onCopy: (value: string, feedbackKey?: string) => Promise<void>;
  onPromptBisqueAuthentication: (message: string) => Promise<void>;
  onOpenConversationFilesInViewer: (fileIds: string[]) => void;
  onImportBisqueResourcesIntoConversation: (
    resourcesToImport: string[],
    options?: {
      materialize?: boolean;
      persistSelectionContext?: boolean;
      source?: string;
      suggestedDomain?: string | null;
      suggestedToolNames?: string[];
      originatingMessageId?: string | null;
      originatingUserText?: string | null;
    }
  ) => Promise<BisqueImportedSelection>;
  onCopyBisqueResourceUri: (resourceUri: string) => Promise<void>;
  onEditUserMessage: (content: string) => void;
  onDeleteUserMessage: (messageId: string) => void;
};

type ConversationMessageRowProps = {
  message: UiMessage;
  isLastMessage: boolean;
  isStreamingAssistant: boolean;
  copiedMessageId: string | null;
  conversationRunArtifacts: RunImageArtifact[];
  uploadedFiles: UploadedFileRecord[];
  bisqueLinksByFileId: Record<string, BisqueViewerLink>;
  apiClient: ApiClient;
  actions: ConversationTranscriptActions;
};

const ConversationMessageRow = memo(
  function ConversationMessageRow({
    message,
    isLastMessage,
    isStreamingAssistant,
    copiedMessageId,
    conversationRunArtifacts,
    uploadedFiles,
    bisqueLinksByFileId,
    apiClient,
    actions,
  }: ConversationMessageRowProps) {
    const isAssistant = message.role === "assistant";
    const isCopied = copiedMessageId === message.id;
    const proModeDevCopyKey = `pro-mode-dev-${message.id}`;
    const isProModeDevCopied = copiedMessageId === proModeDevCopyKey;
    const progressEvents = message.progressEvents ?? EMPTY_PROGRESS_EVENTS;
    const runEvents = message.runEvents ?? EMPTY_RUN_EVENTS;
    const runArtifacts = message.runArtifacts ?? EMPTY_RUN_IMAGE_ARTIFACTS;
    const artifactReferences = useMemo(
      () => mergeRunImageArtifacts(runArtifacts, conversationRunArtifacts),
      [conversationRunArtifacts, runArtifacts]
    );
    const displayContent = useMemo(
      () => rewriteArtifactMarkdownImageUrls(message.content, artifactReferences),
      [artifactReferences, message.content]
    );
    const uploadPreviewUrlForFile = useCallback(
      (fileId: string) => apiClient.uploadPreviewUrl(fileId),
      [apiClient]
    );
    const toolResultCards = useMemo(
      () =>
        buildToolResultCards(
          progressEvents,
          runArtifacts,
          uploadedFiles,
          uploadPreviewUrlForFile,
          {
            runId: message.runId,
            buildArtifactDownloadUrl: (runId, path) =>
              apiClient.artifactDownloadUrl(runId, path),
            responseMetadata: message.responseMetadata ?? null,
          }
        ),
      [
        apiClient,
        message.responseMetadata,
        message.runId,
        progressEvents,
        runArtifacts,
        uploadedFiles,
        uploadPreviewUrlForFile,
      ]
    );
    const leadingToolResultCards = useMemo(
      () => toolResultCards.filter((card) => card.placement === "before_text"),
      [toolResultCards]
    );
    const trailingToolResultCards = useMemo(
      () => toolResultCards.filter((card) => card.placement !== "before_text"),
      [toolResultCards]
    );
    const researchDigest = useMemo(
      () =>
        isAssistant
          ? buildResearchDigestData({
              message,
              hasToolCards: toolResultCards.length > 0,
              hasScientificPrimary: toolResultCards.some((card) => Boolean(card.megasegInsights)),
            })
          : null,
      [isAssistant, message, toolResultCards]
    );
    const hasScientificPrimaryToolCard = toolResultCards.some((card) =>
      Boolean(card.megasegInsights)
    );
    const collapseScientificInterpretation = useMemo(() => {
      if (isStreamingAssistant || !hasScientificPrimaryToolCard) {
        return false;
      }
      const content = String(message.content || "").trim();
      if (!content) {
        return false;
      }
      return (
        content.length > 320 ||
        /(^|\n)#{1,6}\s+/m.test(content) ||
        /(^|\n)\|.+\|/m.test(content) ||
        /(^|\n)(?:- |\d+\. )/m.test(content)
      );
    }, [hasScientificPrimaryToolCard, isStreamingAssistant, message.content]);
    const hasPrimaryToolCard = toolResultCards.length > 0;
    const showResearchDigest =
      Boolean(researchDigest) && (!isStreamingAssistant || !message.liveStream);
    const showLeadingToolResultCards =
      leadingToolResultCards.length > 0 && (!isStreamingAssistant || !message.liveStream);
    const showTrailingToolResultCards =
      trailingToolResultCards.length > 0 && (!isStreamingAssistant || !message.liveStream);
    const bisqueAuthGate = useMemo(
      () => extractBisqueAuthGate(progressEvents),
      [progressEvents]
    );
    const thinkingBarText = useMemo(
      () => thinkingBarTextForRunEvents(runEvents, isStreamingAssistant),
      [isStreamingAssistant, runEvents]
    );
    const proModeDevConversation = useMemo(
      () => proModeDevConversationForMessage(message),
      [message]
    );
    const reasonedDurationLabel = useMemo(
      () => formatReasoningDuration(message.durationSeconds),
      [message.durationSeconds]
    );
    const summaryModeLabel = useMemo(
      () => summaryModeLabelForMessage(message),
      [message]
    );
    const tokenUsage = useMemo(() => extractRunTokenUsage(message), [message]);
    const showAssistantMetadataLine =
      Boolean(reasonedDurationLabel) || Boolean(summaryModeLabel) || Boolean(tokenUsage);
    if (!isAssistant) {
      return (
        <Message
          className="chat-width-frame mx-auto w-full justify-end px-4 sm:px-6"
        >
          <div className="group flex w-full flex-col items-end gap-1">
            <MessageContent className="max-w-full bg-muted text-primary rounded-3xl px-5 py-2.5">
              {message.content}
            </MessageContent>
            {message.uploadedFileNames?.length ? (
              <p className="panel-caption">
                Attached: {message.uploadedFileNames.join(", ")}
              </p>
            ) : null}
            <MessageActions
              className={cn(
                "gap-0 opacity-0 transition-opacity duration-150 group-hover:opacity-100"
              )}
            >
              <MessageAction tooltip="Edit">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  className="rounded-full"
                  onClick={() => actions.onEditUserMessage(message.content)}
                >
                  <Pencil className="size-4" />
                </Button>
              </MessageAction>
              <MessageAction tooltip="Delete">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  className="rounded-full"
                  onClick={() => actions.onDeleteUserMessage(message.id)}
                >
                  <Trash className="size-4" />
                </Button>
              </MessageAction>
              <MessageAction tooltip={isCopied ? "Copied!" : "Copy"}>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  className={cn(
                    "rounded-full transition-all duration-200",
                    isCopied &&
                      "bg-emerald-500/10 text-emerald-600 hover:bg-emerald-500/15 dark:text-emerald-400"
                  )}
                  onClick={() => void actions.onCopy(message.content, message.id)}
                >
                  {isCopied ? (
                    <Check className="size-4 animate-in zoom-in-50 fade-in-0 duration-200" />
                  ) : (
                    <Copy className="size-4" />
                  )}
                </Button>
              </MessageAction>
            </MessageActions>
          </div>
        </Message>
      );
    }

    return (
      <Message
        className="chat-width-frame mx-auto w-full justify-start px-4 sm:px-6"
      >
        <div className="group flex w-full flex-1 flex-col gap-2">
          {showAssistantMetadataLine ? (
            <div className="text-muted-foreground flex flex-wrap items-center gap-2 text-xs leading-5">
              {reasonedDurationLabel ? (
                <span>{`Reasoned for ${reasonedDurationLabel}`}</span>
              ) : null}
              {summaryModeLabel ? (
                <span className="text-[11px] font-medium tracking-[0.02em] text-sky-600/70 dark:text-sky-300/68">
                  {summaryModeLabel}
                </span>
              ) : null}
              {tokenUsage ? (
                <span
                  className="tabular-nums"
                  title={`${tokenUsage.input_tokens.toLocaleString()} input · ${tokenUsage.output_tokens.toLocaleString()} output${
                    tokenUsage.model ? ` · ${tokenUsage.model}` : ""
                  }`}
                >
                  {`${formatTokens(tokenUsage.total_tokens)} tokens`}
                </span>
              ) : null}
            </div>
          ) : null}
          {thinkingBarText ? (
            <div className="mb-1">
              <ThinkingBar
                text={thinkingBarText}
                onStop={actions.onStopConversation}
                stopLabel="Stop"
              />
            </div>
          ) : null}
          {isStreamingAssistant ? (
            <Suspense fallback={null}>
              <LazyChatRunSteps
                runEvents={runEvents}
                progressEvents={progressEvents}
                isStreaming={isStreamingAssistant}
                fallbackLabel={thinkingBarText}
              />
            </Suspense>
          ) : null}
          {showLeadingToolResultCards ? (
            <Suspense fallback={null}>
              <LazyToolResultCardSection
                cards={leadingToolResultCards}
                messageId={message.id}
                onImportBisqueResourcesIntoConversation={
                  actions.onImportBisqueResourcesIntoConversation
                }
                onCopyBisqueResourceUri={actions.onCopyBisqueResourceUri}
              />
            </Suspense>
          ) : null}
          {isStreamingAssistant && message.liveStream ? (
            <div
              id={message.id}
              className="pk-message-content rounded-lg bg-transparent p-0 text-foreground break-words whitespace-normal"
            >
              <MarkdownResponseStream
                className="w-full bg-transparent p-0 text-foreground"
                textStream={message.liveStream}
                onComplete={() => actions.onStreamingRenderComplete(message.id)}
              />
            </div>
          ) : (
            collapseScientificInterpretation ? (
              <details className="chat-scientific-appendix">
                <summary>Model interpretation</summary>
                <div className="chat-scientific-appendix-body">
                  <MessageContent
                    className="w-full bg-transparent p-0 text-foreground"
                    id={message.id}
                    markdown
                  >
                    {displayContent}
                  </MessageContent>
                </div>
              </details>
            ) : (
              <MessageContent
                className="w-full bg-transparent p-0 text-foreground"
                id={message.id}
                markdown
              >
                {displayContent}
              </MessageContent>
            )
          )}
          {proModeDevConversation ? (
            <Suspense fallback={null}>
              <LazyProModeDevTrace
                messageId={message.id}
                conversation={proModeDevConversation}
                isCopied={isProModeDevCopied}
                onCopy={(copyText: string) =>
                  void actions.onCopy(copyText, proModeDevCopyKey)
                }
              />
            </Suspense>
          ) : null}
          {bisqueAuthGate ? (
            <div
              className="mt-3 flex flex-col gap-3 rounded-xl border border-amber-300/70 bg-amber-50/80 p-4 text-sm text-amber-950 shadow-sm backdrop-blur"
              data-testid="bisque-auth-required"
            >
              <div className="flex flex-col gap-1">
                <strong className="text-sm font-semibold">
                  BisQue sign-in required
                </strong>
                <p className="m-0">{bisqueAuthGate.message}</p>
                {bisqueAuthGate.selectedTools.length > 0 ? (
                  <p className="m-0 text-xs text-amber-900/80">
                    Needed tools: {bisqueAuthGate.selectedTools.join(", ")}
                  </p>
                ) : null}
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  size="sm"
                  onClick={() => {
                    void actions.onPromptBisqueAuthentication(
                      bisqueAuthGate.message
                    );
                  }}
                >
                  Sign in to BisQue
                </Button>
              </div>
            </div>
          ) : null}
          {message.quickPreviewFileIds &&
          message.quickPreviewFileIds.length > 0 &&
          !hasPrimaryToolCard ? (
            <Suspense fallback={null}>
              <LazyInlineDataQuickPreview
                fileIds={message.quickPreviewFileIds}
                uploadedFiles={uploadedFiles}
                bisqueLinksByFileId={bisqueLinksByFileId}
                apiClient={apiClient}
                onOpenInViewer={actions.onOpenConversationFilesInViewer}
              />
            </Suspense>
          ) : null}
          {showTrailingToolResultCards ? (
            <Suspense fallback={null}>
              <LazyToolResultCardSection
                cards={trailingToolResultCards}
                messageId={message.id}
                onImportBisqueResourcesIntoConversation={
                  actions.onImportBisqueResourcesIntoConversation
                }
                onCopyBisqueResourceUri={actions.onCopyBisqueResourceUri}
              />
            </Suspense>
          ) : null}
          {showResearchDigest && researchDigest ? (
            <Suspense fallback={null}>
              <LazyResearchDigestCard
                digest={researchDigest}
                followsVisuals={showLeadingToolResultCards}
              />
            </Suspense>
          ) : null}
          {runArtifacts.length > 0 && toolResultCards.length === 0 ? (
            <div className="chat-artifact-grid">
              {runArtifacts.map((artifact) => (
                <a
                  key={artifact.path}
                  href={artifact.downloadUrl ?? artifact.url}
                  target="_blank"
                  rel="noreferrer"
                  className="chat-artifact-card"
                >
                  {artifact.previewable ? (
                    <img
                      src={artifact.url}
                      alt={artifact.title}
                      loading="lazy"
                      className="chat-artifact-image"
                    />
                  ) : (
                    <div className="chat-artifact-image chat-tool-image-placeholder">
                      <ImageIcon className="size-5" />
                      <span>Preview unavailable</span>
                    </div>
                  )}
                  <span className="chat-artifact-title">{artifact.title}</span>
                </a>
              ))}
            </div>
          ) : null}
          <MessageActions
            className={cn(
              "-ml-2.5 gap-0 opacity-0 transition-opacity duration-150 group-hover:opacity-100",
              isLastMessage && "opacity-100"
            )}
          >
            <MessageAction tooltip={isCopied ? "Copied!" : "Copy"}>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className={cn(
                  "rounded-full transition-all duration-200",
                  isCopied &&
                    "bg-emerald-500/10 text-emerald-600 hover:bg-emerald-500/15 dark:text-emerald-400"
                )}
                onClick={() => void actions.onCopy(message.content, message.id)}
              >
                {isCopied ? (
                  <Check className="size-4 animate-in zoom-in-50 fade-in-0 duration-200" />
                ) : (
                  <Copy className="size-4" />
                )}
              </Button>
            </MessageAction>
            <MessageAction tooltip="Upvote">
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="rounded-full"
              >
                <ThumbsUp className="size-4" />
              </Button>
            </MessageAction>
            <MessageAction tooltip="Downvote">
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="rounded-full"
              >
                <ThumbsDown className="size-4" />
              </Button>
            </MessageAction>
          </MessageActions>
        </div>
      </Message>
    );
  },
  (previousProps, nextProps) =>
    previousProps.message === nextProps.message &&
    previousProps.isLastMessage === nextProps.isLastMessage &&
    previousProps.isStreamingAssistant === nextProps.isStreamingAssistant &&
    previousProps.copiedMessageId === nextProps.copiedMessageId &&
    previousProps.conversationRunArtifacts === nextProps.conversationRunArtifacts &&
    previousProps.uploadedFiles === nextProps.uploadedFiles &&
    previousProps.bisqueLinksByFileId === nextProps.bisqueLinksByFileId &&
    previousProps.apiClient === nextProps.apiClient
);

type ConversationTranscriptProps = {
  conversationHydrated: boolean;
  messages: UiMessage[];
  blankChatTokenUsage: TokenUsageResponse | null;
  blankChatUsageLoading: boolean;
  blankChatUsageError: string | null;
  streamingMessageId: string | null;
  copiedMessageId: string | null;
  uploadedFiles: UploadedFileRecord[];
  bisqueLinksByFileId: Record<string, BisqueViewerLink>;
  apiClient: ApiClient;
  actions: ConversationTranscriptActions;
};

const ConversationTranscript = memo(
  function ConversationTranscript({
    conversationHydrated,
    messages,
    blankChatTokenUsage,
    blankChatUsageLoading,
    blankChatUsageError,
    streamingMessageId,
    copiedMessageId,
    uploadedFiles,
    bisqueLinksByFileId,
    apiClient,
    actions,
  }: ConversationTranscriptProps) {
    const conversationRunArtifacts = useMemo(
      () => collectConversationRunArtifacts(messages),
      [messages]
    );
    return (
      <ChatContainerContent
        className="space-y-0 px-4 py-8 sm:px-6 sm:py-14"
        scrollClassName="h-full min-h-0 overscroll-y-contain"
      >
        {!conversationHydrated ? (
          <div className="hero-state">
            <h2 className="hero-title">Loading chat…</h2>
            <p className="hero-subtitle">
              Restoring the full conversation only when you open it to keep memory usage low.
            </p>
          </div>
        ) : messages.length === 0 ? (
          <div className="blank-chat-usage-state">
            <UserTokenUsagePanel
              tokenUsage={blankChatTokenUsage}
              loading={blankChatUsageLoading}
              error={blankChatUsageError}
              className="blank-chat-usage-panel"
              density="compact"
            />
          </div>
        ) : (
          messages.map((message, index) => (
            <ConversationMessageRow
              key={message.id}
              message={message}
              isLastMessage={index === messages.length - 1}
              isStreamingAssistant={streamingMessageId === message.id}
              copiedMessageId={copiedMessageId}
              conversationRunArtifacts={conversationRunArtifacts}
              uploadedFiles={uploadedFiles}
              bisqueLinksByFileId={bisqueLinksByFileId}
              apiClient={apiClient}
              actions={actions}
            />
          ))
        )}
      </ChatContainerContent>
    );
  },
  (previousProps, nextProps) =>
    previousProps.conversationHydrated === nextProps.conversationHydrated &&
    previousProps.messages === nextProps.messages &&
    previousProps.blankChatTokenUsage === nextProps.blankChatTokenUsage &&
    previousProps.blankChatUsageLoading === nextProps.blankChatUsageLoading &&
    previousProps.blankChatUsageError === nextProps.blankChatUsageError &&
    previousProps.streamingMessageId === nextProps.streamingMessageId &&
    previousProps.copiedMessageId === nextProps.copiedMessageId &&
    previousProps.uploadedFiles === nextProps.uploadedFiles &&
    previousProps.bisqueLinksByFileId === nextProps.bisqueLinksByFileId &&
    previousProps.apiClient === nextProps.apiClient
);

const toChatWire = (messages: UiMessage[]): ChatMessage[] =>
  messages.map((message) => ({ role: message.role, content: message.content }));

const summarizePrompt = (value: string, maxLen = 46): string => {
  const singleLine = value.replace(/\s+/g, " ").trim();
  if (singleLine.length <= maxLen) {
    return singleLine;
  }
  return `${singleLine.slice(0, maxLen - 1)}…`;
};

const summarizeConversationTitle = (value: string, maxWords = 6): string => {
  return fallbackConversationTitleFromText(value, maxWords);
};

const normalizeConversationTitle = (value: string): string => {
  const singleLine = value.replace(/\s+/g, " ").trim().replace(/^["'`]+|["'`]+$/g, "");
  if (!singleLine) {
    return "New conversation";
  }
  if (singleLine.length <= 120) {
    return singleLine;
  }
  return `${singleLine.slice(0, 119)}…`;
};

const getPeriodLabel = (timestamp: number): HistoryPeriod => {
  const now = new Date();
  const then = new Date(timestamp);

  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfThen = new Date(then.getFullYear(), then.getMonth(), then.getDate());
  const diffDays = Math.floor(
    (startOfToday.getTime() - startOfThen.getTime()) / (24 * 60 * 60 * 1000)
  );

  if (diffDays <= 0) {
    return "Today";
  }
  if (diffDays === 1) {
    return "Yesterday";
  }
  if (diffDays < 7) {
    return "Last 7 days";
  }
  return "Older";
};

const cleanupMatchedUrl = (rawUrl: string): string =>
  rawUrl.replace(/[)\],.;>]+$/g, "");

const isBisqueResourceUrl = (url: string): boolean =>
  /\/client_service\/view\?resource=/i.test(url) ||
  /\/data_service\/[^/?#\s]+/i.test(url) ||
  /\/image_service\/[^/?#\s]+/i.test(url);

const extractBisqueUrls = (text: string): string[] => {
  if (!text) {
    return [];
  }
  const matches = text.match(/https?:\/\/[^\s<>"'`]+/gi) ?? [];
  const filtered = matches
    .map((candidate) => cleanupMatchedUrl(candidate))
    .filter((candidate) => isBisqueResourceUrl(candidate));
  return [...new Set(filtered)];
};

const stripBisqueUrls = (text: string): string => {
  if (!text) {
    return "";
  }
  const matches = extractBisqueUrls(text);
  if (matches.length === 0) {
    return text.trim();
  }
  let next = text;
  matches.forEach((url) => {
    next = next.split(url).join(" ");
  });
  return next.replace(/\s+/g, " ").trim();
};

const isBisqueDatasetUri = (value: string): boolean => {
  const normalized = decodeURIComponent(String(value || "").trim().toLowerCase());
  return /\/data_service\/dataset(?:\/|$|\?)/.test(normalized);
};

const partitionBisqueUris = (
  uris: Array<string | null | undefined>,
  datasetUriHints: Array<string | null | undefined> = []
): { resourceUris: string[]; datasetUris: string[] } => {
  const datasetHintSet = new Set(
    datasetUriHints
      .map((entry) => String(entry || "").trim().toLowerCase())
      .filter((entry) => entry.length > 0)
  );
  const resourceUris: string[] = [];
  const datasetUris: string[] = [];
  const seenResources = new Set<string>();
  const seenDatasets = new Set<string>();
  uris.forEach((entry) => {
    const uri = String(entry || "").trim();
    if (!uri) {
      return;
    }
    const key = uri.toLowerCase();
    if (datasetHintSet.has(key) || isBisqueDatasetUri(uri)) {
      if (!seenDatasets.has(key)) {
        seenDatasets.add(key);
        datasetUris.push(uri);
      }
      return;
    }
    if (!seenResources.has(key)) {
      seenResources.add(key);
      resourceUris.push(uri);
    }
  });
  return { resourceUris, datasetUris };
};

const toBisqueClientViewUrl = (urlValue: string | null | undefined): string | null => {
  const candidate = String(urlValue || "").trim();
  if (!candidate) {
    return null;
  }
  const preferredRoot =
    inferBisqueRootFromUrl(DEFAULT_BISQUE_BROWSER_URL) || inferBisqueRootFromUrl(candidate);
  if (!preferredRoot) {
    return /\/client_service\/view\?resource=/i.test(candidate) ? candidate : null;
  }
  if (/\/client_service\/view\?resource=/i.test(candidate)) {
    try {
      const parsed =
        typeof window !== "undefined" && window.location?.origin
          ? new URL(candidate, window.location.origin)
          : new URL(candidate);
      const resourceValue = String(parsed.searchParams.get("resource") || "").trim();
      if (!resourceValue) {
        return candidate;
      }
      const normalizedResource = toBisqueClientViewUrl(resourceValue);
      if (!normalizedResource) {
        return candidate;
      }
      const normalizedParsed =
        typeof window !== "undefined" && window.location?.origin
          ? new URL(normalizedResource, window.location.origin)
          : new URL(normalizedResource);
      const normalizedResourceUri = String(
        normalizedParsed.searchParams.get("resource") || resourceValue
      ).trim();
      return `${preferredRoot}/client_service/view?resource=${normalizedResourceUri}`;
    } catch {
      return candidate;
    }
  }
  try {
    const parsed =
      typeof window !== "undefined" && window.location?.origin
        ? new URL(candidate, window.location.origin)
        : new URL(candidate);
    const normalizedPath = parsed.pathname.replace("/image_service/", "/data_service/");
    if (/\/data_service\//i.test(normalizedPath)) {
      return `${preferredRoot}/client_service/view?resource=${preferredRoot}${normalizedPath}`;
    }
  } catch {
    if (/\/image_service\//i.test(candidate)) {
      return `${preferredRoot}/client_service/view?resource=${preferredRoot}${candidate.replace("/image_service/", "/data_service/")}`;
    }
    if (/\/data_service\//i.test(candidate)) {
      return `${preferredRoot}/client_service/view?resource=${preferredRoot}${candidate}`;
    }
  }
  return null;
};

const normalizeApiError = (error: unknown): string => {
  if (error instanceof ApiError) {
    if (
      error.detail &&
      typeof error.detail === "object" &&
      !Array.isArray(error.detail)
    ) {
      const detail = error.detail as Record<string, unknown>;
      const detailMessage = String(detail.message ?? detail.error ?? "").trim();
      if (detailMessage) {
        return detailMessage;
      }
    }
    if (typeof error.detail === "string") {
      return `${error.message}: ${error.detail}`;
    }
    return `${error.message}: ${JSON.stringify(error.detail)}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
};

const accountApprovalMessageFromSession = (
  session: Pick<
    BisqueAuthSessionResponse,
    "account_status" | "account_email" | "message" | "authenticated"
  >
): string | null => {
  const explicitMessage = String(session.message ?? "").trim();
  if (explicitMessage) {
    return explicitMessage;
  }
  const status = String(session.account_status ?? "").trim().toLowerCase();
  const accountEmail = String(session.account_email ?? "").trim();
  const accountLabel = accountEmail ? ` for ${accountEmail}` : "";
  if (status === "pending") {
    return `Your account request${accountLabel} is pending administrator approval.`;
  }
  if (status === "disabled") {
    return `This account${accountLabel} has been disabled. Contact an administrator for access.`;
  }
  if (status === "rejected") {
    return `This account request${accountLabel} was not approved. Contact an administrator for details.`;
  }
  if (status && !session.authenticated) {
    return `This account${accountLabel} is ${status} and cannot access the platform yet.`;
  }
  return null;
};

const isBisqueAuthApiError = (error: unknown): boolean => {
  if (!(error instanceof ApiError)) {
    return false;
  }
  if (![401, 403].includes(error.status)) {
    return false;
  }
  const detail =
    typeof error.detail === "string" ? error.detail : JSON.stringify(error.detail ?? {});
  return /bisque|sign-?in|session credentials/i.test(`${error.message} ${detail}`);
};

const isAbortError = (error: unknown): boolean => {
  return (
    (error instanceof DOMException && error.name === "AbortError") ||
    (error instanceof Error && error.name === "AbortError")
  );
};

const isTransientStreamTransportError = (error: unknown, message: string): boolean => {
  if (error instanceof ApiError) {
    return false;
  }
  if (isAbortError(error)) {
    return false;
  }
  const normalized = message.toLowerCase();
  return (
    normalized.includes("load failed") ||
    normalized.includes("failed to fetch") ||
    normalized.includes("network request failed") ||
    normalized.includes("networkerror") ||
    normalized.includes("the network connection was lost") ||
    normalized.includes("terminated")
  );
};

const artifactImageExtensions = [
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".bmp",
  ".svg",
  ".avif",
  ".tif",
  ".tiff",
  ".nii",
  ".nii.gz",
  ".nrrd",
  ".mha",
  ".mhd",
];

const artifactInlineImageExtensions = [
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".bmp",
  ".svg",
  ".avif",
];

const artifactPreviewPrefixPatterns = [
  /^[a-f0-9]{32}__[a-f0-9]{12}__/i,
  /^[a-f0-9-]{36}__[a-f0-9]{12}__/i,
  /^[a-f0-9]{32}__/i,
  /^[a-f0-9-]{36}__/i,
  /^[a-f0-9]{12}__/i,
];

const isImageArtifactPath = (path: string): boolean => {
  const lower = path.toLowerCase();
  return artifactImageExtensions.some((ext) => lower.endsWith(ext));
};

const isInlineImageArtifactPath = (path: string): boolean => {
  const lower = path.toLowerCase();
  return artifactInlineImageExtensions.some((ext) => lower.endsWith(ext));
};

const isInlineImageArtifact = (
  path: string,
  mimeType?: string | null
): boolean => {
  const normalizedMime = String(mimeType ?? "").toLowerCase();
  if (normalizedMime) {
    if (normalizedMime.startsWith("image/tif")) {
      return false;
    }
    if (normalizedMime.startsWith("image/")) {
      return true;
    }
  }
  return isInlineImageArtifactPath(path);
};

const stripArtifactFilenamePrefixes = (value: string): string => {
  let normalized = extractFilename(value).trim();
  if (!normalized) {
    return "";
  }
  let changed = true;
  while (changed && normalized) {
    changed = false;
    for (const pattern of artifactPreviewPrefixPatterns) {
      const next = normalized.replace(pattern, "");
      if (next !== normalized) {
        normalized = next;
        changed = true;
      }
    }
  }
  return normalized;
};

const artifactTitleFromPath = (path: string): string => {
  const lower = path.toLowerCase();
  if (lower.includes("side_by_side")) {
    return "Original + mask";
  }
  if (lower.includes("overlay")) {
    return "Mask overlay";
  }
  if (lower.includes("mask_preview")) {
    return "Mask preview";
  }
  const dehashed = stripArtifactFilenamePrefixes(path);
  if (dehashed.length <= 64) {
    return dehashed;
  }
  return `${dehashed.slice(0, 28)}…${dehashed.slice(-22)}`;
};

const artifactSourceNameFromPath = (path: string): string => {
  return stripArtifactFilenamePrefixes(path);
};

const artifactDisplayName = (artifact: Pick<ArtifactRecord, "path" | "title" | "source_path">): string => {
  const titled = String(artifact.title || "").trim();
  if (titled) {
    return titled;
  }
  const sourcePath = String(artifact.source_path || "").trim();
  if (sourcePath) {
    return extractFilename(sourcePath);
  }
  return artifactSourceNameFromPath(artifact.path);
};

const isIntermediateTileArtifactPath = (path: string): boolean => {
  return /-\d{4}-x\d+-y\d+(?=\.[^.]+$)/i.test(path);
};

const artifactHydrationPriority = (
  artifact: Pick<ArtifactRecord, "path" | "title" | "source_path">
): [number, string] => {
  const path = String(artifact.path || "").trim();
  const haystack = `${path} ${String(artifact.title || "")} ${String(
    artifact.source_path || ""
  )}`.toLowerCase();
  if (path.startsWith("uploads/")) {
    return [0, path];
  }
  if (haystack.includes("matplotlib_annotated")) {
    return [1, path];
  }
  if (path.startsWith("tool_outputs/raw/")) {
    return [2, path];
  }
  if (isIntermediateTileArtifactPath(path)) {
    return [4, path];
  }
  return [3, path];
};

const prioritizeHydratedImageArtifacts = (
  artifacts: ArtifactRecord[],
  limit = 60
): ArtifactRecord[] => {
  return [...artifacts]
    .sort((left, right) => {
      const leftKey = artifactHydrationPriority(left);
      const rightKey = artifactHydrationPriority(right);
      return leftKey[0] - rightKey[0] || leftKey[1].localeCompare(rightKey[1]);
    })
    .slice(0, limit);
};

const isPlotArtifact = (
  artifact:
    | Pick<ArtifactRecord, "path" | "title">
    | Pick<RunImageArtifact, "path" | "title" | "sourceName">
): boolean => {
  const haystack = [
    String(artifact.path || ""),
    String(artifact.title || ""),
    "sourceName" in artifact ? String(artifact.sourceName || "") : "",
  ]
    .join(" ")
    .toLowerCase();
  return /(class_distribution|confidence|histogram|distribution|low_confidence|review|outlier|tail)/i.test(
    haystack
  );
};

const formatPercentMetric = (value: number | null | undefined, digits = 2): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return `${Number(value).toFixed(digits)}%`;
};

const formatIntegerMetric = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return Math.round(Number(value)).toLocaleString();
};

const scientificFigureSortKey = (
  value: Pick<ScientificFigureCard, "title" | "subtitle" | "summary">
): [number, string] => {
  const normalized = [value.title, value.subtitle, value.summary]
    .filter((item): item is string => Boolean(item))
    .join(" ")
    .toLowerCase();
  if (/\bmip\b|maximum/i.test(normalized)) {
    return [0, normalized];
  }
  if (/mid[-\s]?z|midplane|representative/i.test(normalized)) {
    return [1, normalized];
  }
  return [2, normalized];
};

const eventArtifactToToolImage = ({
  artifact,
  runId,
  buildArtifactDownloadUrl,
}: {
  artifact: Record<string, unknown>;
  runId?: string;
  buildArtifactDownloadUrl?: (runId: string, path: string) => string;
}): ToolCardImage | null => {
  const path = String(artifact.path ?? "").trim();
  if (!path || !isImageArtifactPath(path) || !runId || !buildArtifactDownloadUrl) {
    return null;
  }
  const mimeType = String(artifact.mime_type ?? "").trim() || undefined;
  const downloadUrl = buildArtifactDownloadUrl(runId, path);
  const title =
    String(artifact.title ?? artifactDisplayName({ path, title: "", source_path: String(artifact.source_path ?? "") }))
      .trim() || artifactTitleFromPath(path);
  return {
    path,
    url: downloadUrl,
    downloadUrl,
    title,
    sourceName: title,
    sourcePath: String(artifact.source_path ?? "").trim() || undefined,
    previewable: isInlineImageArtifact(path, mimeType),
  } satisfies ToolCardImage;
};

const buildMegasegNarrative = ({
  fileRows,
  processed,
  meanCoverage,
  meanObjectCount,
}: {
  fileRows: MegasegFileInsight[];
  processed: number | null;
  meanCoverage: number | null;
  meanObjectCount: number | null;
}): string | undefined => {
  const firstRow = fileRows[0];
  const technicalSummary = String(firstRow?.technicalSummary ?? "").trim();
  if (technicalSummary) {
    return technicalSummary;
  }

  const processedCount = Math.max(
    fileRows.length,
    Math.round(processed ?? fileRows.length)
  );
  if (processedCount <= 0) {
    return undefined;
  }

  const coverageClause =
    meanCoverage !== null
      ? `${formatPercentMetric(meanCoverage)} mean segmented coverage`
      : "a completed segmentation pass";
  const objectClause =
    meanObjectCount !== null ? ` with ${meanObjectCount.toFixed(1)} mean objects per image` : "";
  return `${coverageClause}${objectClause} across ${pluralizeCount(processedCount, "image")}. Review the overlays first to confirm boundary fidelity, then use the table to compare coverage and object counts across inputs.`;
};

const artifactLookupKeys = (value: string): string[] => {
  const rawName = extractFilename(value).toLowerCase().trim();
  if (!rawName) {
    return [];
  }
  const keys = new Set<string>();

  const pushVariant = (candidate: string): void => {
    const normalized = candidate.trim().toLowerCase();
    if (!normalized) {
      return;
    }

    const derivedVariants = [
      normalized,
      normalized.replace(/__preview(?=\.[^.]+$)/i, ""),
      normalized.replace(/_{1,2}det-[a-z0-9_-]+(?=\.[^.]+$)/i, ""),
      normalized.replace(/-0000-x\d+-y\d+(?=\.[^.]+$)/i, ""),
      normalized.replace(/\.[^.]+$/, ""),
      normalized.replace(/^([a-f0-9]{12,36})_([a-f0-9]{12,36})_(.+)$/i, "$1__$2_$3"),
      normalized.replace(/^([a-f0-9]{12,36})__([a-f0-9]{12,36})_(.+)$/i, "$1_$2_$3"),
    ];

    derivedVariants
      .map((item) => item.trim())
      .filter((item) => item.length > 0)
      .forEach((item) => keys.add(item));
  };

  pushVariant(rawName);
  pushVariant(stripArtifactFilenamePrefixes(rawName));
  pushVariant(stripArtifactFilenamePrefixes(value));

  return Array.from(keys);
};

const extractFilename = (value: string): string => {
  const normalized = String(value || "").replace(/\\/g, "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] ?? normalized;
};

const toDisplayFileLabel = (value: string): string => {
  const dehashed = stripArtifactFilenamePrefixes(value);
  if (dehashed.length <= 64) {
    return dehashed;
  }
  return `${dehashed.slice(0, 28)}…${dehashed.slice(-22)}`;
};

const buildUploadedArtifactPreviewLookup = (
  uploadedFiles: UploadedFileRecord[]
): Map<string, UploadedFileRecord[]> => {
  const lookup = new Map<string, UploadedFileRecord[]>();
  uploadedFiles
    .filter((file) => isImageLikeUploadedFile(file))
    .forEach((file) => {
      artifactLookupKeys(file.original_name).forEach((key) => {
        const existing = lookup.get(key) ?? [];
        if (!existing.some((item) => item.file_id === file.file_id)) {
          existing.push(file);
        }
        lookup.set(key, existing);
      });
    });
  return lookup;
};

const resolveUploadedArtifactPreview = (
  artifactPath: string,
  uploadedPreviewLookup: Map<string, UploadedFileRecord[]>
): UploadedFileRecord | null => {
  for (const key of artifactLookupKeys(artifactPath)) {
    const match = uploadedPreviewLookup.get(key)?.[0];
    if (match) {
      return match;
    }
  }
  return null;
};

const normalizeToolName = (value: unknown): string => {
  if (typeof value !== "string") {
    return "";
  }
  return value.trim();
};

const toNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
};

const toRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
};

const generatedConversationTitleFromResponse = (
  response: ChatResponse | null | undefined
): string | null => {
  const metadata = toRecord(response?.metadata);
  const title = normalizeConversationTitle(String(metadata?.conversation_title ?? ""));
  return title === "New conversation" ? null : title;
};

const shouldApplyGeneratedConversationTitle = (
  currentTitle: string,
  generatedTitle: string,
  temporaryTitle?: string
): boolean => {
  const current = normalizeConversationTitle(currentTitle);
  if (current === generatedTitle) {
    return false;
  }
  if (current === "New conversation") {
    return true;
  }
  return Boolean(
    temporaryTitle && current === normalizeConversationTitle(temporaryTitle)
  );
};

const humanizeScientificLabel = (value: string): string => {
  const normalized = String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!normalized) {
    return "Value";
  }
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
};

const formatScientificScalar = (value: unknown): string => {
  if (typeof value === "number" && Number.isFinite(value)) {
    if (Number.isInteger(value)) {
      return value.toLocaleString();
    }
    return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  const numeric = toNumber(value);
  if (numeric !== null) {
    return formatScientificScalar(numeric);
  }
  const text = String(value ?? "").trim();
  return text || "n/a";
};

const formatContractMeasurement = (record: Record<string, unknown>): ResearchDigestMeasurementRow | null => {
  const rawName = String(record.name ?? record.metric ?? record.label ?? "").trim();
  const name = humanizeScientificLabel(rawName || "measurement");
  const value = record.value ?? record.result ?? record.measurement;
  const unit = String(record.unit ?? "").trim();
  const ci95 = Array.isArray(record.ci95) ? record.ci95 : null;
  const ciLabel =
    ci95 &&
    ci95.length === 2 &&
    ci95.every((item) => toNumber(item) !== null)
      ? ` (95% CI ${formatScientificScalar(ci95[0])} to ${formatScientificScalar(ci95[1])}${unit ? ` ${unit}` : ""})`
      : "";
  const baseValue = `${formatScientificScalar(value)}${unit ? ` ${unit}` : ""}`.trim();
  if (!name || !baseValue) {
    return null;
  }
  return {
    name,
    valueLabel: `${baseValue}${ciLabel}`.trim(),
  };
};

const summarizeStatisticalRecord = (record: Record<string, unknown>): ResearchDigestStatisticRow | null => {
  const preferredLabel =
    String(record.summary ?? record.test ?? record.method ?? record.name ?? record.metric ?? "")
      .trim();
  const label = humanizeScientificLabel(preferredLabel || "analysis");
  const pieces: string[] = [];
  [
    ["comparison", record.comparison],
    ["finding", record.finding],
    ["p", record.p_value],
    ["effect", record.effect_size],
    ["n", record.n],
    ["statistic", record.statistic],
    ["notes", record.notes],
  ].forEach(([key, value]) => {
    if (value === null || value === undefined || value === "") {
      return;
    }
    pieces.push(`${humanizeScientificLabel(String(key))} ${formatScientificScalar(value)}`);
  });
  const summary = pieces.join(" · ").trim() || String(record.summary ?? "").trim();
  if (!summary) {
    return null;
  }
  return { label, summary };
};

const coerceAssistantContract = (value: unknown): AssistantContract | null => {
  const record = toRecord(value);
  if (!record) {
    return null;
  }
  const result = String(record.result ?? "").trim();
  const evidence = Array.isArray(record.evidence)
    ? record.evidence.map((item) => toRecord(item)).filter((item): item is Record<string, unknown> => item !== null)
    : [];
  const measurements = Array.isArray(record.measurements)
    ? record.measurements
        .map((item) => toRecord(item))
        .filter((item): item is Record<string, unknown> => item !== null)
    : [];
  const statisticalAnalysisRaw = record.statistical_analysis;
  const statisticalAnalysis = Array.isArray(statisticalAnalysisRaw)
    ? statisticalAnalysisRaw
        .map((item) => toRecord(item))
        .filter((item): item is Record<string, unknown> => item !== null)
    : [];
  const confidenceRecord = toRecord(record.confidence);
  const qcWarnings = Array.isArray(record.qc_warnings)
    ? record.qc_warnings.map((item) => String(item || "").trim()).filter(Boolean)
    : [];
  const limitations = Array.isArray(record.limitations)
    ? record.limitations.map((item) => String(item || "").trim()).filter(Boolean)
    : [];
  const nextSteps = Array.isArray(record.next_steps)
    ? record.next_steps
        .map((item) => {
          if (typeof item === "string") {
            return { action: item };
          }
          const step = toRecord(item);
          return step ? { action: String(step.action ?? "").trim() } : null;
        })
        .filter((item): item is { action: string } => Boolean(item?.action))
    : [];
  if (
    !result &&
    evidence.length === 0 &&
    measurements.length === 0 &&
    statisticalAnalysis.length === 0 &&
    qcWarnings.length === 0 &&
    limitations.length === 0 &&
    nextSteps.length === 0
  ) {
    return null;
  }
  return {
    result,
    evidence: evidence as AssistantContract["evidence"],
    measurements: measurements as AssistantContract["measurements"],
    statistical_analysis: statisticalAnalysis,
    confidence: {
      level:
        String(confidenceRecord?.level ?? "").trim().toLowerCase() === "high"
          ? "high"
          : String(confidenceRecord?.level ?? "").trim().toLowerCase() === "low"
            ? "low"
            : "medium",
      why: Array.isArray(confidenceRecord?.why)
        ? confidenceRecord.why.map((item) => String(item || "").trim()).filter(Boolean)
        : [],
    },
    qc_warnings: qcWarnings,
    limitations,
    next_steps: nextSteps,
  };
};

const extractAssistantContractFromMessage = (message: UiMessage): AssistantContract | null => {
  const metadata = toRecord(message.responseMetadata);
  const contractCandidates: unknown[] = [
    metadata?.contract,
    toRecord(metadata?.pro_mode)?.contract,
    toRecord(toRecord(metadata?.ui_hydrated)?.contract),
  ];
  for (const candidate of contractCandidates) {
    const contract = coerceAssistantContract(candidate);
    if (contract) {
      return contract;
    }
  }
  for (const event of [...(message.progressEvents ?? [])].reverse()) {
    if (String(event.event || "").trim().toLowerCase() !== "workpad_contract") {
      continue;
    }
    const contract = coerceAssistantContract(event.contract);
    if (contract) {
      return contract;
    }
  }
  return null;
};

const buildResearchDigestData = ({
  message,
  hasToolCards,
  hasScientificPrimary,
}: {
  message: UiMessage;
  hasToolCards: boolean;
  hasScientificPrimary: boolean;
}): ResearchDigestData | null => {
  const contract = extractAssistantContractFromMessage(message);
  if (!contract) {
    return null;
  }
  const metadata = toRecord(message.responseMetadata);
  const proMode = toRecord(metadata?.pro_mode);
  const executionPath = String(proMode?.execution_path ?? "").trim().toLowerCase();
  const toolInvocations = Array.isArray(metadata?.tool_invocations)
    ? metadata.tool_invocations
        .map((item) => toRecord(item))
        .filter((item): item is Record<string, unknown> => item !== null)
    : [];
  const hasToolBackedContext =
    hasToolCards ||
    toolInvocations.length > 0 ||
    executionPath === "tool_workflow" ||
    executionPath === "research_program";
  const evidence = contract.evidence
    .map((item) => toRecord(item))
    .filter((item): item is Record<string, unknown> => item !== null)
    .map((item) => ({
      source: humanizeScientificLabel(String(item.source ?? "evidence").trim() || "evidence"),
      summary: String(item.summary ?? "").trim() || undefined,
      artifact: String(item.artifact ?? "").trim() || undefined,
      runId: String(item.run_id ?? "").trim() || null,
    }))
    .filter((item) => item.source || item.summary)
    .slice(0, 4);
  const measurements = contract.measurements
    .map((item) => toRecord(item))
    .filter((item): item is Record<string, unknown> => item !== null)
    .map((item) => formatContractMeasurement(item))
    .filter((item): item is ResearchDigestMeasurementRow => item !== null)
    .slice(0, 6);
  const statisticalAnalysis = contract.statistical_analysis
    .map((item) => summarizeStatisticalRecord(item))
    .filter((item): item is ResearchDigestStatisticRow => item !== null)
    .slice(0, 4);
  const qcWarnings = contract.qc_warnings.slice(0, 4);
  const limitations = contract.limitations.slice(0, 4);
  const nextSteps = contract.next_steps
    .map((item) => String(item.action ?? "").trim())
    .filter(Boolean)
    .slice(0, 4);
  const result = String(contract.result || "").trim() || String(message.content || "").trim();
  const hasQuantitativeEvidence = measurements.length > 0 || statisticalAnalysis.length > 0;
  const populatedSections = [
    evidence.length > 0,
    measurements.length > 0,
    statisticalAnalysis.length > 0,
    qcWarnings.length > 0,
    limitations.length > 0,
    nextSteps.length > 0,
  ].filter(Boolean).length;
  if (!result) {
    return null;
  }
  if (!hasToolBackedContext || !hasQuantitativeEvidence) {
    return null;
  }
  if (hasScientificPrimary) {
    return null;
  }
  if (!hasToolCards && populatedSections < 2 && result.length < 120) {
    return null;
  }
  return {
    result,
    confidenceLevel: contract.confidence?.level,
    confidenceWhy: Array.isArray(contract.confidence?.why) ? contract.confidence.why.slice(0, 2) : [],
    evidence,
    measurements,
    statisticalAnalysis,
    qcWarnings,
    limitations,
    nextSteps,
  };
};

const runEventIdentity = (event: RunEvent): string =>
  JSON.stringify({
    event_type: String(event.event_type || "").trim().toLowerCase(),
    level: String(event.level || "").trim().toLowerCase(),
    payload: toRecord(event.payload) ?? event.payload ?? null,
  });

const appendUniqueRunEvent = (events: RunEvent[], nextEvent: RunEvent): RunEvent[] => {
  const nextIdentity = runEventIdentity(nextEvent);
  if (events.some((event) => runEventIdentity(event) === nextIdentity)) {
    return events;
  }
  return [...events, nextEvent];
};

const extractBisqueAuthGate = (
  progressEvents: ProgressEvent[]
): { message: string; selectedTools: string[] } | null => {
  for (const event of progressEvents) {
    const summary = toRecord(event.summary);
    if (!summary) {
      continue;
    }
    if (String(summary.error_code ?? "").trim().toLowerCase() !== "bisque_auth_required") {
      continue;
    }
    const selectedTools = Array.isArray(summary.selected_tools)
      ? summary.selected_tools
          .map((item) => String(item ?? "").trim())
          .filter((item) => item.length > 0)
      : [];
    return {
      message:
        String(event.message ?? "").trim() ||
        "BisQue authentication is required for this request.",
      selectedTools,
    };
  }
  return null;
};

const isPrairieDetectionClassName = (className: string): boolean => {
  const normalized = String(className || "").trim().toLowerCase();
  return normalized === "prairie_dog" || normalized === "burrow";
};

const toToolDetectionBox = (value: unknown): ToolDetectionBox | null => {
  const row = toRecord(value);
  if (!row) {
    return null;
  }
  const className = String(row.class_name ?? row.class ?? "").trim();
  const xyxy = Array.isArray(row.xyxy) ? row.xyxy : [];
  if (!className || xyxy.length < 4) {
    return null;
  }
  const coordinates = xyxy
    .slice(0, 4)
    .map((entry) => toNumber(entry))
    .filter((entry): entry is number => entry !== null);
  if (coordinates.length < 4) {
    return null;
  }
  const [x1, y1, x2, y2] = coordinates;
  const xMin = Math.min(x1, x2);
  const yMin = Math.min(y1, y2);
  const xMax = Math.max(x1, x2);
  const yMax = Math.max(y1, y2);
  if (xMax <= xMin || yMax <= yMin) {
    return null;
  }
  return {
    className,
    confidence: toNumber(row.confidence),
    xMin,
    yMin,
    xMax,
    yMax,
  };
};

const toPrairieImageAnalysis = (value: unknown): PrairieImageAnalysis | null => {
  const row = toRecord(value);
  if (!row) {
    return null;
  }
  const rawFile = String(
    row.rawFile ?? row.raw_file ?? row.path ?? row.file ?? row.source_path ?? ""
  ).trim();
  if (!rawFile) {
    return null;
  }
  const prairieBurrowContext = toRecord(row.prairie_burrow_context);
  const geo = toRecord(row.geo);
  const latitude = toNumber(geo?.latitude);
  const longitude = toNumber(geo?.longitude);
  return {
    rawFile,
    fileLabel: String(row.fileLabel ?? row.file_label ?? "").trim() || toDisplayFileLabel(rawFile),
    prairieDogCount:
      toNumber(prairieBurrowContext?.prairie_dog_count) ??
      toNumber(toRecord(row.class_counts)?.prairie_dog),
    burrowCount:
      toNumber(prairieBurrowContext?.burrow_count) ??
      toNumber(toRecord(row.class_counts)?.burrow),
    boxCount: toNumber(row.box_count),
    nearestBurrowDistancePxMean: toNumber(prairieBurrowContext?.nearest_burrow_distance_px_mean),
    nearestBurrowDistancePxMin: toNumber(prairieBurrowContext?.nearest_burrow_distance_px_min),
    nearestBurrowDistancePxMedian: toNumber(
      prairieBurrowContext?.nearest_burrow_distance_px_median
    ),
    nearestBurrowDistancePxMax: toNumber(prairieBurrowContext?.nearest_burrow_distance_px_max),
    overlappingBurrowCount: toNumber(
      prairieBurrowContext?.prairie_dogs_overlapping_burrows
    ),
    capturedAt: String(row.captured_at ?? "").trim() || undefined,
    latitude,
    longitude,
  };
};

const buildToolResultCards = (
  progressEvents: ProgressEvent[],
  runArtifacts: RunImageArtifact[],
  uploadedFiles: UploadedFileRecord[] = [],
  buildUploadPreviewUrl: (fileId: string) => string = (fileId) =>
    `/v2/uploads/${encodeURIComponent(fileId)}/preview`,
  options?: {
    runId?: string;
    buildArtifactDownloadUrl?: (runId: string, path: string) => string;
    responseMetadata?: Record<string, unknown> | null;
  }
): ToolResultCard[] => {
  if (!progressEvents.length) {
    return [];
  }
  type BisqueSearchCandidate = {
    index: number;
    toolName: "search_bisque_resources" | "bisque_advanced_search";
    matchCount: number | null;
    resourceType?: string;
    resourceRows: ToolResourceRow[];
  };

  const artifactBySource = new Map<string, RunImageArtifact[]>();
  const toolInvocationSummaryByTool = new Map<string, Record<string, unknown>>();
  const uploadedPreviewLookup = buildUploadedArtifactPreviewLookup(uploadedFiles);
  const responseMetadataRecord = toRecord(options?.responseMetadata);
  const toolInvocations = Array.isArray(responseMetadataRecord?.tool_invocations)
    ? responseMetadataRecord.tool_invocations
    : [];
  const toolInvocationRecords = toolInvocations
    .map((entry) => toRecord(entry))
    .filter((entry): entry is Record<string, unknown> => entry !== null);
  const scientificToolInvocations = toolInvocationRecords.map((entry) => ({
    tool: String(entry.tool ?? ""),
    status: String(entry.status ?? ""),
    output_summary: toRecord(entry.output_summary),
    output_envelope: toRecord(entry.output_envelope),
  }));
  toolInvocationRecords.forEach((entry) => {
      const tool = normalizeToolName(entry.tool);
      const outputSummary = toRecord(entry.output_summary);
      if (!tool || !outputSummary) {
        return;
      }
      toolInvocationSummaryByTool.set(tool, outputSummary);
    });
  runArtifacts.forEach((artifact) => {
    const lookupValues = new Set<string>([
      artifact.sourceName,
      artifact.path,
      artifact.title,
      artifact.sourcePath ?? "",
    ]);
    lookupValues.forEach((lookupValue) => {
      artifactLookupKeys(lookupValue).forEach((key) => {
        const existing = artifactBySource.get(key) ?? [];
        if (!existing.some((item) => item.path === artifact.path)) {
          existing.push(artifact);
        }
        artifactBySource.set(key, existing);
      });
    });
  });

  const cards: ToolResultCard[] = [];
  const bisqueSearchByType = new Map<string, BisqueSearchCandidate>();
  const bisqueMetadataByResource = new Map<
    string,
    { index: number; infoScore: number; card: ToolResultCard }
  >();
  const bisqueDownloadRows: ToolDownloadRow[] = [];
  let latestBisqueDownloadIndex = -1;
  let latestBisqueUploadCard: ToolResultCard | null = null;
  let bestBisqueFindAssetsCard: ToolResultCard | null = null;
  let bestBisqueFindAssetsScore = Number.NEGATIVE_INFINITY;
  let bestBisqueFindAssetsIndex = -1;
  let hasSuccessfulBisqueModule = false;
  const scientificResultGroups = buildScientificResultGroups({
    progressEvents,
    toolInvocations: scientificToolInvocations,
    runArtifacts: runArtifacts.map((artifact) => ({
      path: artifact.path,
      title: artifact.title,
      sourcePath: artifact.sourcePath,
      url: artifact.url,
      downloadUrl: artifact.downloadUrl,
      previewable: artifact.previewable,
      resultGroupId: artifact.resultGroupId ?? null,
    })),
    runId: options?.runId,
    buildArtifactDownloadUrl: options?.buildArtifactDownloadUrl,
  });
  scientificResultGroups.forEach((group, groupIndex) => {
    const singleFile = group.fileRows.length <= 1;
    const firstRow = group.fileRows[0];
    cards.push({
      id: `scientific-result-${group.resultGroupId}-${groupIndex}`,
      tool: "segment_image_megaseg",
      title:
        group.fileRows.length > 1 ? "Megaseg cohort summary" : "Megaseg segmentation",
      metrics: singleFile
        ? [
            {
              label: "Coverage",
              value: formatPercentMetric(group.metrics.coveragePercent),
            },
            {
              label: "Objects",
              value: formatIntegerMetric(group.metrics.objectCount),
            },
            {
              label: "Active z-slices",
              value:
                group.metrics.activeSliceCount !== null &&
                group.metrics.activeSliceCount !== undefined &&
                group.metrics.zSliceCount !== null &&
                group.metrics.zSliceCount !== undefined
                  ? `${formatIntegerMetric(group.metrics.activeSliceCount)}/${formatIntegerMetric(group.metrics.zSliceCount)}`
                  : "n/a",
            },
            {
              label: "Largest component",
              value: formatIntegerMetric(group.metrics.largestComponentVoxels),
            },
          ]
        : [
            {
              label: "Images",
              value: `${group.fileRows.length}`,
            },
            {
              label: "Mean coverage",
              value: formatPercentMetric(group.metrics.coveragePercent),
            },
            {
              label: "Mean objects",
              value: formatIntegerMetric(group.metrics.objectCount),
            },
            {
              label: "Representative figures",
              value: `${1 + group.secondaryFigures.length}`,
            },
          ],
      classes: [],
      images: [],
      resourceRows: [],
      downloadRows: [],
      placement: "before_text",
      narrative:
        group.technicalSummary ??
        buildMegasegNarrative({
          fileRows: group.fileRows,
          processed: group.fileRows.length,
          meanCoverage: group.metrics.coveragePercent,
          meanObjectCount: group.metrics.objectCount,
        }),
      megasegInsights: {
        resultGroupId: group.resultGroupId,
        heroFigure: group.heroFigure,
        secondaryFigures: group.secondaryFigures,
        fileRows: group.fileRows,
        reportPath: group.reportPath ?? null,
        summaryCsvPath: group.summaryCsvPath ?? null,
        reportDownloadUrl:
          group.reportPath && options?.runId && options?.buildArtifactDownloadUrl
            ? options.buildArtifactDownloadUrl(options.runId, group.reportPath)
            : null,
        summaryCsvDownloadUrl:
          group.summaryCsvPath && options?.runId && options?.buildArtifactDownloadUrl
            ? options.buildArtifactDownloadUrl(options.runId, group.summaryCsvPath)
            : null,
        technicalSummary: group.technicalSummary ?? null,
        collectionLabel:
          group.fileRows.length > 1
            ? `${group.fileRows.length} images summarized in one report-first result`
            : firstRow?.file
              ? toDisplayFileLabel(firstRow.file)
              : undefined,
        device: null,
        structureChannel: null,
        nucleusChannel: null,
      },
    });
  });
  const hasMegasegScientificResultSurface = scientificResultGroups.length > 0;
  const mergeBisqueResourceRows = (
    left: ToolResourceRow[],
    right: ToolResourceRow[]
  ): ToolResourceRow[] => {
    const merged = new Map<string, ToolResourceRow>();
    [...left, ...right].forEach((row) => {
      const key =
        row.resourceUri?.toLowerCase() ||
        row.clientViewUrl?.toLowerCase() ||
        row.uri?.toLowerCase() ||
        `${row.name.toLowerCase()}|${String(row.owner ?? "").toLowerCase()}|${String(
          row.created ?? ""
        ).toLowerCase()}|${String(row.resourceType ?? "").toLowerCase()}`;
      if (!merged.has(key)) {
        merged.set(key, row);
      }
    });
    return Array.from(merged.values()).slice(0, 12);
  };
  progressEvents.forEach((event, index) => {
    if (event.event !== "completed") {
      return;
    }
    const toolName = normalizeToolName(event.tool);
    if (
      toolName !== "segment_image_megaseg" &&
      toolName !== "yolo_detect" &&
      toolName !== "estimate_depth_pro" &&
      toolName !== "quantify_segmentation_masks" &&
      toolName !== "plot_quantified_detections" &&
      toolName !== "upload_to_bisque" &&
      toolName !== "load_bisque_resource" &&
      toolName !== "bisque_download_resource" &&
      toolName !== "bisque_download_dataset" &&
      toolName !== "bisque_create_dataset" &&
      toolName !== "bisque_add_to_dataset" &&
      toolName !== "bisque_add_gobjects" &&
      toolName !== "add_tags_to_resource" &&
      toolName !== "bisque_fetch_xml" &&
      toolName !== "delete_bisque_resource" &&
      toolName !== "run_bisque_module" &&
      toolName !== "search_bisque_resources" &&
      toolName !== "bisque_advanced_search" &&
      toolName !== "bisque_find_assets"
    ) {
      return;
    }

    if (
      hasMegasegScientificResultSurface &&
      (toolName === "segment_image_megaseg" ||
        toolName === "quantify_segmentation_masks")
    ) {
      return;
    }

    const summary = toRecord(event.summary);
    const artifacts = Array.isArray(event.artifacts)
      ? event.artifacts.map((item) => toRecord(item)).filter((item): item is Record<string, unknown> => item !== null)
      : [];
    const matchedImages: RunImageArtifact[] = [];
    const matchedImageByIdentity = new Map<string, number>();

    const addMatchedImage = (artifact: RunImageArtifact): void => {
      if (matchedImages.some((item) => item.path === artifact.path)) {
        return;
      }
      const identityKeys = artifactLookupKeys(artifact.sourceName);
      const identity =
        identityKeys[identityKeys.length - 1] ??
        artifact.sourceName.toLowerCase();
      const existingIndex = matchedImageByIdentity.get(identity);
      if (existingIndex === undefined) {
        matchedImageByIdentity.set(identity, matchedImages.length);
        matchedImages.push(artifact);
        return;
      }
      if (!matchedImages[existingIndex].previewable && artifact.previewable) {
        matchedImages[existingIndex] = artifact;
      }
    };

    artifacts.forEach((artifact) => {
      const sourcePath = String(artifact.path ?? "");
      if (!sourcePath) {
        return;
      }
      artifactLookupKeys(sourcePath).forEach((key) => {
        const matches = artifactBySource.get(key) ?? [];
        matches.forEach((match) => addMatchedImage(match));
      });
    });

    if (toolName === "run_bisque_module") {
      if (summary?.success === false) {
        return;
      }
      hasSuccessfulBisqueModule = true;
      bisqueSearchByType.clear();
      bestBisqueFindAssetsCard = null;
      bestBisqueFindAssetsScore = Number.NEGATIVE_INFINITY;
      bestBisqueFindAssetsIndex = -1;

      if (matchedImages.length === 0) {
        runArtifacts
          .filter((artifact) =>
            /(edge|canny|module|module_output|preview|output)/i.test(
              artifact.sourceName
            )
          )
          .slice(0, 6)
          .forEach((artifact) => addMatchedImage(artifact));
      }

      const moduleName =
        typeof summary?.module_name === "string" && summary.module_name.trim()
          ? summary.module_name.trim()
          : "Module";
      const status =
        typeof summary?.status === "string" && summary.status.trim()
          ? summary.status.trim()
          : "completed";
      const outputPath =
        typeof summary?.output_path === "string" && summary.output_path.trim()
          ? summary.output_path.trim()
          : "";
      const outputResourceUri =
        typeof summary?.output_resource_uri === "string" &&
        summary.output_resource_uri.trim()
          ? summary.output_resource_uri.trim()
          : "";
      const outputClientViewUrl =
        typeof summary?.output_client_view_url === "string" &&
        summary.output_client_view_url.trim()
          ? summary.output_client_view_url.trim()
          : "";
      const outputImageServiceUrl =
        typeof summary?.output_image_service_url === "string" &&
        summary.output_image_service_url.trim()
          ? summary.output_image_service_url.trim()
          : "";
      const downloadedOutput = Boolean(summary?.downloaded_output) || Boolean(outputPath);

      const hasInlinePreview = matchedImages.some((image) => image.previewable);
      if (outputImageServiceUrl && !hasInlinePreview) {
        if (!matchedImages.some((image) => image.url === outputImageServiceUrl)) {
          matchedImages.push({
            path: outputImageServiceUrl,
            url: outputImageServiceUrl,
            title: `${moduleName} output`,
            sourceName: extractFilename(outputImageServiceUrl),
            previewable: true,
          });
        }
      }

      const resourceRows: ToolResourceRow[] = [];
      if (outputClientViewUrl || outputResourceUri) {
        resourceRows.push({
          name: `${moduleName} output`,
          resourceType: "image",
          uri: outputClientViewUrl || outputResourceUri,
          resourceUri: outputResourceUri || undefined,
          clientViewUrl: outputClientViewUrl || undefined,
          imageServiceUrl: outputImageServiceUrl || undefined,
        });
      }
      if (outputImageServiceUrl && outputImageServiceUrl !== outputClientViewUrl) {
        resourceRows.push({
          name: "Image service URL",
          resourceType: "image_service",
          uri: outputImageServiceUrl,
          resourceUri: outputResourceUri || undefined,
          clientViewUrl: outputClientViewUrl || undefined,
          imageServiceUrl: outputImageServiceUrl,
        });
      }

      cards.push({
        id: `${toolName}-${index}`,
        tool: "run_bisque_module",
        title: "BisQue module",
        subtitle: moduleName,
        metrics: [
          {
            label: "Status",
            value: status,
          },
          {
            label: "Output",
            value: downloadedOutput ? "downloaded" : "resource link",
          },
          {
            label: "Images",
            value: `${matchedImages.length}`,
          },
        ],
        classes: [],
        images: [...matchedImages]
          .sort((left, right) => Number(right.previewable) - Number(left.previewable))
          .slice(0, 6),
        resourceRows,
        downloadRows: [],
      });
      return;
    }

    if (
      toolName === "search_bisque_resources" ||
      toolName === "bisque_advanced_search" ||
      toolName === "bisque_find_assets"
    ) {
      if (summary?.success === false) {
        return;
      }
      if (hasSuccessfulBisqueModule) {
        return;
      }
      const summaryRows = Array.isArray(summary?.rows) ? summary.rows : [];
      const resourceRows = summaryRows
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => {
          const resourceUri = String(row.resource_uri ?? "").trim();
          const clientViewUrl = String(row.client_view_url ?? "").trim();
          const imageServiceUrl = String(row.image_service_url ?? "").trim();
          const uri = String(row.uri ?? clientViewUrl ?? resourceUri ?? "").trim();
          const rawName = String(row.name ?? "").trim();
          return {
            name:
              rawName ||
              (uri ? toDisplayFileLabel(uri.split("/").pop() || uri) : "resource"),
            owner: String(row.owner ?? "").trim() || undefined,
            created: String(row.created ?? "").trim() || undefined,
            resourceType: String(row.resource_type ?? "").trim() || undefined,
            uri: uri || undefined,
            resourceUri: resourceUri || undefined,
            clientViewUrl: clientViewUrl || undefined,
            imageServiceUrl: imageServiceUrl || undefined,
          };
        })
        .slice(0, 12);
      const downloadRows = (
        Array.isArray(summary?.download_rows) ? summary.download_rows : []
      )
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          status: String(row.status ?? "unknown").trim() || "unknown",
          outputPath: String(row.output_path ?? "").trim() || undefined,
          resourceUri: String(row.resource_uri ?? "").trim() || undefined,
          clientViewUrl: String(row.client_view_url ?? "").trim() || undefined,
          imageServiceUrl: String(row.image_service_url ?? "").trim() || undefined,
          error: String(row.error ?? "").trim() || undefined,
        }))
        .slice(0, 12);
      const matchCount = toNumber(summary?.count);
      const resourceType =
        typeof summary?.resource_type === "string" && summary.resource_type.trim()
          ? summary.resource_type.trim()
          : undefined;
      const metadataLoaded = toNumber(summary?.metadata_loaded);
      const downloadsTotal = toNumber(summary?.downloads_total);
      const downloadsSuccess = toNumber(summary?.downloads_success);

      if (toolName !== "bisque_find_assets") {
        const resourceTypeKey = (resourceType || "resource").toLowerCase();
        const candidate: BisqueSearchCandidate = {
          index,
          toolName:
            toolName === "bisque_advanced_search"
              ? "bisque_advanced_search"
              : "search_bisque_resources",
          matchCount,
          resourceType,
          resourceRows,
        };
        const existing = bisqueSearchByType.get(resourceTypeKey);
        if (!existing) {
          bisqueSearchByType.set(resourceTypeKey, candidate);
          return;
        }
        const mergedRows = mergeBisqueResourceRows(existing.resourceRows, candidate.resourceRows);
        const mergedCount = Math.max(
          mergedRows.length,
          existing.matchCount ?? 0,
          candidate.matchCount ?? 0
        );
        bisqueSearchByType.set(resourceTypeKey, {
          index: Math.max(existing.index, candidate.index),
          toolName:
            candidate.toolName === "bisque_advanced_search" ||
            existing.toolName === "bisque_advanced_search"
              ? "bisque_advanced_search"
              : "search_bisque_resources",
          matchCount: mergedCount,
          resourceType: candidate.resourceType ?? existing.resourceType,
          resourceRows: mergedRows,
        });
        return;
      }

      const hasMeaningfulFindAssetsResult =
        resourceRows.length > 0 ||
        downloadRows.length > 0 ||
        (matchCount ?? 0) > 0 ||
        (metadataLoaded ?? 0) > 0 ||
        (downloadsTotal ?? 0) > 0 ||
        (downloadsSuccess ?? 0) > 0;
      if (!hasMeaningfulFindAssetsResult) {
        return;
      }

      const findAssetsCard: ToolResultCard = {
        id: `${toolName}-${index}`,
        tool: "bisque_find_assets",
        title: "BisQue assets",
        subtitle: resourceType ? `${resourceType} resources` : undefined,
        metrics: [
          {
            label: "Matches",
            value:
              matchCount !== null
                ? `${Math.round(matchCount)}`
                : `${resourceRows.length}`,
          },
          {
            label: "Shown",
            value: `${resourceRows.length}`,
          },
          {
            label: "Metadata",
            value:
              metadataLoaded !== null ? `${Math.round(metadataLoaded)}` : "0",
          },
          {
            label: "Downloads",
            value:
              downloadsTotal !== null
                ? `${Math.round(downloadsSuccess ?? 0)}/${Math.round(downloadsTotal)}`
                : "0",
          },
        ],
        classes: [],
        images: toolCardImagesFromBisqueResourceRows(resourceRows, 1),
        resourceRows,
        downloadRows,
      };
      const findAssetsScore =
        (matchCount ?? resourceRows.length) * 1000 +
        resourceRows.length * 10 +
        (metadataLoaded ?? 0) +
        (downloadsSuccess ?? 0);
      if (
        findAssetsScore > bestBisqueFindAssetsScore ||
        (findAssetsScore === bestBisqueFindAssetsScore &&
          index > bestBisqueFindAssetsIndex)
      ) {
        bestBisqueFindAssetsScore = findAssetsScore;
        bestBisqueFindAssetsIndex = index;
        bestBisqueFindAssetsCard = findAssetsCard;
      }
      return;
    }

    if (toolName === "load_bisque_resource") {
      if (summary?.success === false) {
        return;
      }
      const summaryRows = Array.isArray(summary?.rows) ? summary.rows : [];
      const resourceRows = summaryRows
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          name: String(row.name ?? "").trim() || "resource",
          owner: String(row.owner ?? "").trim() || undefined,
          created: String(row.created ?? "").trim() || undefined,
          resourceType: String(row.resource_type ?? "").trim() || undefined,
          uri: String(row.uri ?? "").trim() || undefined,
          resourceUri: String(row.resource_uri ?? "").trim() || undefined,
          clientViewUrl: String(row.client_view_url ?? "").trim() || undefined,
          imageServiceUrl: String(row.image_service_url ?? "").trim() || undefined,
        }))
        .slice(0, 1);
      const dimensions = toRecord(summary?.dimensions);
      const dimensionLabel = dimensions
        ? ([
            ["w", toNumber(dimensions.width)],
            ["h", toNumber(dimensions.height)],
            ["z", toNumber(dimensions.depth)],
            ["c", toNumber(dimensions.channels)],
            ["t", toNumber(dimensions.timepoints)],
          ] as Array<[string, number | null]>)
            .filter(([, value]) => value !== null)
            .map(([label, value]) => `${label}=${Math.round(value ?? 0)}`)
            .join(", ")
        : "";
      const tagCount = Math.round(toNumber(summary?.tag_count) ?? 0);

      const metadataCard: ToolResultCard = {
        id: `${toolName}-${index}`,
        tool: "load_bisque_resource",
        title: "BisQue metadata",
        subtitle: resourceRows[0]?.name,
        metrics: [
          {
            label: "Tags",
            value: `${tagCount}`,
          },
          {
            label: "Dimensions",
            value: dimensionLabel || "n/a",
          },
        ],
        classes: [],
        images: toolCardImagesFromBisqueResourceRows(resourceRows, 1),
        resourceRows,
        downloadRows: [],
      };
      const metadataKey =
        resourceRows[0]?.resourceUri?.toLowerCase() ||
        resourceRows[0]?.clientViewUrl?.toLowerCase() ||
        resourceRows[0]?.uri?.toLowerCase() ||
        resourceRows[0]?.name?.toLowerCase() ||
        `${toolName}-${index}`;
      const infoScore =
        tagCount * 10 +
        Number(Boolean(dimensionLabel)) * 5 +
        Number(resourceRows.length > 0) * 2 +
        Number(metadataCard.images.length > 0);
      const existingMetadata = bisqueMetadataByResource.get(metadataKey);
      if (
        !existingMetadata ||
        infoScore > existingMetadata.infoScore ||
        (infoScore === existingMetadata.infoScore && index > existingMetadata.index)
      ) {
        bisqueMetadataByResource.set(metadataKey, {
          index,
          infoScore,
          card: metadataCard,
        });
      }
      return;
    }

    if (toolName === "upload_to_bisque") {
      const summaryRows = Array.isArray(summary?.rows) ? summary.rows : [];
      const resourceRows = summaryRows
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          name: String(row.name ?? "").trim() || "uploaded resource",
          owner: String(row.owner ?? "").trim() || undefined,
          created: String(row.created ?? "").trim() || undefined,
          resourceType: String(row.resource_type ?? "").trim() || undefined,
          uri: String(row.uri ?? "").trim() || undefined,
          resourceUri: String(row.resource_uri ?? "").trim() || undefined,
          clientViewUrl: String(row.client_view_url ?? "").trim() || undefined,
          imageServiceUrl: String(row.image_service_url ?? "").trim() || undefined,
        }))
        .slice(0, 12);
      const uploaded = Math.max(0, Math.round(toNumber(summary?.uploaded) ?? 0));
      const total = Math.max(uploaded, Math.round(toNumber(summary?.total) ?? uploaded));
      const hasUsableUploadResult =
        summary?.success !== false || resourceRows.length > 0 || uploaded > 0;
      if (!hasUsableUploadResult) {
        return;
      }
      const datasetAction =
        typeof summary?.dataset_action === "string" && summary.dataset_action.trim()
          ? summary.dataset_action.trim()
          : undefined;
      const datasetName =
        typeof summary?.dataset_name === "string" && summary.dataset_name.trim()
          ? summary.dataset_name.trim()
          : undefined;
      const datasetMembersAdded = toNumber(summary?.dataset_members_added);

      latestBisqueUploadCard = {
        id: `${toolName}-${index}`,
        tool: "upload_to_bisque",
        title: "BisQue upload",
        subtitle: datasetName,
        metrics: [
          {
            label: "Uploaded",
            value: `${uploaded}/${total}`,
          },
          {
            label: "Dataset",
            value: datasetAction ?? "none",
          },
          {
            label: "Added",
            value:
              datasetMembersAdded !== null ? `${Math.round(datasetMembersAdded)}` : "0",
          },
        ],
        classes: [],
        images: toolCardImagesFromBisqueResourceRows(resourceRows, 1),
        resourceRows,
        downloadRows: [],
      };
      return;
    }

    if (toolName === "bisque_create_dataset" || toolName === "bisque_add_to_dataset") {
      if (summary?.success === false) {
        return;
      }
      const resourceRows = (Array.isArray(summary?.rows) ? summary.rows : [])
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          name: String(row.name ?? "").trim() || "dataset",
          owner: String(row.owner ?? "").trim() || undefined,
          created: String(row.created ?? "").trim() || undefined,
          resourceType: String(row.resource_type ?? "").trim() || undefined,
          uri: String(row.uri ?? "").trim() || undefined,
          resourceUri: String(row.resource_uri ?? "").trim() || undefined,
          clientViewUrl: String(row.client_view_url ?? "").trim() || undefined,
          imageServiceUrl: String(row.image_service_url ?? "").trim() || undefined,
        }))
        .slice(0, 4);
      const action =
        typeof summary?.action === "string" && summary.action.trim()
          ? summary.action.trim()
          : toolName === "bisque_create_dataset"
            ? "created"
            : "updated";
      const members = toNumber(summary?.members);
      const added = toNumber(summary?.added);
      const totalResources = toNumber(summary?.total_resources);
      cards.push({
        id: `${toolName}-${index}`,
        tool: toolName,
        title: "BisQue dataset",
        subtitle:
          typeof summary?.dataset_name === "string" && summary.dataset_name.trim()
            ? summary.dataset_name.trim()
            : undefined,
        metrics: [
          { label: "Action", value: action },
          {
            label: "Members",
            value:
              members !== null
                ? `${Math.round(members)}`
                : totalResources !== null
                  ? `${Math.round(totalResources)}`
                  : "n/a",
          },
          {
            label: "Added",
            value: added !== null ? `${Math.round(added)}` : "n/a",
          },
        ],
        classes: [],
        images: toolCardImagesFromBisqueResourceRows(resourceRows, 1),
        resourceRows,
        downloadRows: [],
      });
      return;
    }

    if (toolName === "bisque_add_gobjects") {
      if (summary?.success === false) {
        return;
      }
      const resourceRows = (Array.isArray(summary?.rows) ? summary.rows : [])
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          name: String(row.name ?? "").trim() || "annotated resource",
          owner: String(row.owner ?? "").trim() || undefined,
          created: String(row.created ?? "").trim() || undefined,
          resourceType: String(row.resource_type ?? "").trim() || undefined,
          uri: String(row.uri ?? "").trim() || undefined,
          resourceUri: String(row.resource_uri ?? "").trim() || undefined,
          clientViewUrl: String(row.client_view_url ?? "").trim() || undefined,
          imageServiceUrl: String(row.image_service_url ?? "").trim() || undefined,
        }))
        .slice(0, 4);
      const countsByType = toRecord(summary?.counts_by_type);
      const classRows = countsByType
        ? Object.entries(countsByType)
            .map(([name, value]) => ({
              name,
              count: Math.max(0, Math.round(toNumber(value) ?? 0)),
            }))
            .filter((row) => row.count > 0)
            .slice(0, 8)
        : [];
      cards.push({
        id: `${toolName}-${index}`,
        tool: "bisque_add_gobjects",
        title: "BisQue annotations",
        metrics: [
          {
            label: "Added",
            value: `${Math.max(0, Math.round(toNumber(summary?.added_total) ?? 0))}`,
          },
          {
            label: "Types",
            value: `${classRows.length}`,
          },
        ],
        classes: classRows,
        images: toolCardImagesFromBisqueResourceRows(resourceRows, 1),
        resourceRows,
        downloadRows: [],
      });
      return;
    }

    if (toolName === "add_tags_to_resource") {
      if (summary?.success === false) {
        return;
      }
      const resourceRows = (Array.isArray(summary?.rows) ? summary.rows : [])
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          name: String(row.name ?? "").trim() || "tagged resource",
          owner: String(row.owner ?? "").trim() || undefined,
          created: String(row.created ?? "").trim() || undefined,
          resourceType: String(row.resource_type ?? "").trim() || undefined,
          uri: String(row.uri ?? "").trim() || undefined,
          resourceUri: String(row.resource_uri ?? "").trim() || undefined,
          clientViewUrl: String(row.client_view_url ?? "").trim() || undefined,
          imageServiceUrl: String(row.image_service_url ?? "").trim() || undefined,
        }))
        .slice(0, 4);
      cards.push({
        id: `${toolName}-${index}`,
        tool: "add_tags_to_resource",
        title: "BisQue tags",
        metrics: [
          {
            label: "Tags added",
            value: `${Math.max(0, Math.round(toNumber(summary?.tag_count) ?? 0))}`,
          },
        ],
        classes: [],
        images: toolCardImagesFromBisqueResourceRows(resourceRows, 1),
        resourceRows,
        downloadRows: [],
      });
      return;
    }

    if (toolName === "bisque_fetch_xml") {
      if (summary?.success === false) {
        return;
      }
      const resourceRows = (Array.isArray(summary?.rows) ? summary.rows : [])
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          name: String(row.name ?? "").trim() || "xml source",
          owner: String(row.owner ?? "").trim() || undefined,
          created: String(row.created ?? "").trim() || undefined,
          resourceType: String(row.resource_type ?? "").trim() || undefined,
          uri: String(row.uri ?? "").trim() || undefined,
          resourceUri: String(row.resource_uri ?? "").trim() || undefined,
          clientViewUrl: String(row.client_view_url ?? "").trim() || undefined,
          imageServiceUrl: String(row.image_service_url ?? "").trim() || undefined,
        }))
        .slice(0, 4);
      cards.push({
        id: `${toolName}-${index}`,
        tool: "bisque_fetch_xml",
        title: "BisQue XML",
        metrics: [
          {
            label: "Truncated",
            value: summary?.truncated ? "yes" : "no",
          },
          {
            label: "Saved",
            value:
              typeof summary?.saved_path === "string" && summary.saved_path.trim()
                ? "yes"
                : "no",
          },
        ],
        classes: [],
        images: toolCardImagesFromBisqueResourceRows(resourceRows, 1),
        resourceRows,
        downloadRows: [],
      });
      return;
    }

    if (toolName === "delete_bisque_resource") {
      const resourceRows = (Array.isArray(summary?.rows) ? summary.rows : [])
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          name: String(row.name ?? "").trim() || "deleted resource",
          owner: String(row.owner ?? "").trim() || undefined,
          created: String(row.created ?? "").trim() || undefined,
          resourceType: String(row.resource_type ?? "").trim() || undefined,
          uri: String(row.uri ?? "").trim() || undefined,
          resourceUri: String(row.resource_uri ?? "").trim() || undefined,
          clientViewUrl: String(row.client_view_url ?? "").trim() || undefined,
          imageServiceUrl: String(row.image_service_url ?? "").trim() || undefined,
        }))
        .slice(0, 1);
      const datasetCleanup = toRecord(summary?.dataset_cleanup);
      const cleanupFound =
        datasetCleanup && typeof datasetCleanup.found === "number" ? datasetCleanup.found : undefined;
      const cleanupRemoved =
        datasetCleanup && typeof datasetCleanup.removed === "number" ? datasetCleanup.removed : undefined;
      const deletionVerified =
        typeof summary?.deletion_verified === "boolean" ? summary.deletion_verified : undefined;
      const deletionError =
        typeof summary?.error === "string" && summary.error.trim() ? summary.error.trim() : undefined;
      const deleteSucceeded = summary?.success !== false;
      cards.push({
        id: `${toolName}-${index}`,
        tool: "delete_bisque_resource",
        title: deleteSucceeded ? "BisQue deletion" : "BisQue deletion failed",
        subtitle:
          typeof summary?.resource_name === "string" && summary.resource_name.trim()
            ? summary.resource_name.trim()
            : undefined,
        metrics: [
          { label: "Status", value: deleteSucceeded ? "deleted" : "failed" },
          ...(typeof deletionVerified === "boolean"
            ? [{ label: "Verified", value: deletionVerified ? "yes" : "no" }]
            : []),
          ...(typeof cleanupFound === "number" && typeof cleanupRemoved === "number"
            ? [{ label: "Dataset cleanup", value: `${cleanupRemoved}/${cleanupFound}` }]
            : []),
          ...(deletionError ? [{ label: "Error", value: deletionError }] : []),
        ],
        classes: [],
        images: [],
        resourceRows,
        downloadRows: [],
      });
      return;
    }

    if (toolName === "bisque_download_resource") {
      if (summary?.success === false) {
        return;
      }
      const rows = (Array.isArray(summary?.download_rows) ? summary.download_rows : [])
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          status: String(row.status ?? "unknown").trim() || "unknown",
          outputPath: String(row.output_path ?? "").trim() || undefined,
          resourceUri: String(row.resource_uri ?? "").trim() || undefined,
          clientViewUrl: String(row.client_view_url ?? "").trim() || undefined,
          imageServiceUrl: String(row.image_service_url ?? "").trim() || undefined,
          error: String(row.error ?? "").trim() || undefined,
        }));
      if (rows.length === 0) {
        return;
      }
      latestBisqueDownloadIndex = Math.max(latestBisqueDownloadIndex, index);
      rows.forEach((row) => bisqueDownloadRows.push(row));
      return;
    }

    if (toolName === "bisque_download_dataset") {
      if (summary?.success === false) {
        return;
      }
      const rows = (Array.isArray(summary?.download_rows) ? summary.download_rows : [])
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          status: String(row.status ?? "unknown").trim() || "unknown",
          outputPath: String(row.output_path ?? "").trim() || undefined,
          resourceUri: String(row.resource_uri ?? "").trim() || undefined,
          clientViewUrl: String(row.client_view_url ?? "").trim() || undefined,
          imageServiceUrl: String(row.image_service_url ?? "").trim() || undefined,
          error: String(row.error ?? "").trim() || undefined,
        }));
      cards.push({
        id: `${toolName}-${index}`,
        tool: "bisque_download_dataset",
        title: "BisQue dataset download",
        metrics: [
          {
            label: "Downloaded",
            value: `${Math.max(0, Math.round(toNumber(summary?.downloaded) ?? 0))}/${Math.max(0, Math.round(toNumber(summary?.total_members) ?? rows.length))}`,
          },
          {
            label: "Files",
            value: `${rows.length}`,
          },
        ],
        classes: [],
        images: [],
        resourceRows: [],
        downloadRows: rows.slice(0, 12),
      });
      return;
    }

    if (toolName === "segment_image_megaseg") {
      const toolInvocationSummary =
        toolInvocationSummaryByTool.get(toolName) ?? null;
      const hydratedUi = toRecord(toRecord(options?.responseMetadata)?.ui_hydrated);
      const megasegSummary: Record<string, unknown> = {
        ...(toolInvocationSummary ?? {}),
        ...summary,
        scientific_summary:
          summary?.scientific_summary ?? toolInvocationSummary?.scientific_summary,
        visualization_paths:
          summary?.visualization_paths ?? toolInvocationSummary?.visualization_paths,
        files_processed:
          summary?.files_processed ?? toolInvocationSummary?.files_processed,
        aggregate: summary?.aggregate ?? toolInvocationSummary?.aggregate,
      };
      const scientificSummary = toRecord(megasegSummary.scientific_summary);
      const aggregate =
        toRecord(scientificSummary?.aggregate) ??
        toRecord(megasegSummary.aggregate) ??
        {};
      const scientificSummaryFiles = Array.isArray(scientificSummary?.files)
        ? scientificSummary.files
        : [];
      const scientificRows = (
        scientificSummaryFiles.length > 0
          ? scientificSummaryFiles
          : Array.isArray(megasegSummary.files_processed)
            ? megasegSummary.files_processed
            : []
      )
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null);
      const summaryFileRows: MegasegFileInsight[] = scientificRows.map((row) => ({
        file:
          String(row.file ?? "").trim() ||
          String(row.path ?? "").trim() ||
          "image",
        coveragePercent: toNumber(row.coverage_percent),
        objectCount: toNumber(row.object_count),
        activeSliceCount: toNumber(row.active_slice_count),
        zSliceCount: toNumber(row.z_slice_count),
        largestComponentVoxels: toNumber(row.largest_component_voxels),
        technicalSummary: String(row.technical_summary ?? "").trim() || undefined,
      }));
      const hydratedRows = (
        Array.isArray(hydratedUi?.megaseg_file_summaries)
          ? hydratedUi.megaseg_file_summaries
          : []
      )
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => {
          const segmentation = toRecord(row.segmentation);
          return {
            file:
              String(row.file ?? "").trim() ||
              String(row.path ?? "").trim() ||
              "image",
            coveragePercent: toNumber(segmentation?.coverage_percent),
            objectCount: toNumber(segmentation?.object_count),
            activeSliceCount: toNumber(segmentation?.active_slice_count),
            zSliceCount: toNumber(segmentation?.z_slice_count),
            largestComponentVoxels: toNumber(segmentation?.largest_component_voxels),
            technicalSummary: String(row.technical_summary ?? "").trim() || undefined,
          } satisfies MegasegFileInsight;
        });
      const fileRows = Array.from(
        [...summaryFileRows, ...hydratedRows].reduce((map, row) => {
          const key =
            artifactLookupKeys(row.file)[0] || String(row.file || "").toLowerCase();
          const existing = map.get(key);
          map.set(key, {
            ...(existing ?? {}),
            ...row,
          });
          return map;
        }, new Map<string, MegasegFileInsight>())
      ).map(([, row]) => row);

      const computedMeanCoverage =
        fileRows.length > 0
          ? fileRows
              .map((row) => row.coveragePercent)
              .filter((value): value is number => value !== null && value !== undefined)
          : [];
      const computedMeanObjects =
        fileRows.length > 0
          ? fileRows
              .map((row) => row.objectCount)
              .filter((value): value is number => value !== null && value !== undefined)
          : [];
      const meanCoverage =
        toNumber(megasegSummary.mean_coverage_percent) ??
        toNumber(aggregate?.mean_coverage_percent) ??
        (computedMeanCoverage.length > 0
          ? computedMeanCoverage.reduce((sum, value) => sum + value, 0) /
            computedMeanCoverage.length
          : null);
      const meanObjectCount =
        toNumber(megasegSummary.mean_object_count) ??
        toNumber(aggregate?.mean_object_count) ??
        (computedMeanObjects.length > 0
          ? computedMeanObjects.reduce((sum, value) => sum + value, 0) /
            computedMeanObjects.length
          : null);
      const processed =
        toNumber(megasegSummary.processed) ??
        toNumber(aggregate?.processed_files);
      const totalFiles =
        toNumber(megasegSummary.total_files) ??
        toNumber(aggregate?.total_files);

      const visualizationRows = (Array.isArray(megasegSummary.visualization_paths)
        ? megasegSummary.visualization_paths
        : [])
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null);

      const fallbackMegasegImages = artifacts
        .map((artifact) =>
          eventArtifactToToolImage({
            artifact,
            runId: options?.runId,
            buildArtifactDownloadUrl: options?.buildArtifactDownloadUrl,
          })
        )
        .filter((artifact): artifact is ToolCardImage => artifact !== null)
        .filter((artifact) => /(megaseg|overlay|midz|mip)/i.test(`${artifact.path} ${artifact.title}`));

      const rowForValue = (value: string): MegasegFileInsight | undefined => {
        const keys = new Set(artifactLookupKeys(value));
        return fileRows.find((row) =>
          artifactLookupKeys(row.file).some((key) => keys.has(key))
        );
      };

      const figureCards = visualizationRows
        .map((row, vizIndex): ScientificFigureCard | null => {
          const path = String(row.path ?? "").trim();
          if (!path) {
            return null;
          }
          const sourceFile = String(row.file ?? "").trim();
          let displayedArtifact =
            artifactLookupKeys(path)
              .flatMap((key) => artifactBySource.get(key) ?? [])
              .find(Boolean) ?? null;
          if (!displayedArtifact && sourceFile) {
            displayedArtifact =
              artifactLookupKeys(sourceFile)
                .flatMap((key) => artifactBySource.get(key) ?? [])
                .find(Boolean) ?? null;
          }
          const fallbackArtifact =
            displayedArtifact ??
            eventArtifactToToolImage({
              artifact: {
                path,
                title: row.title,
                source_path: sourceFile,
              },
              runId: options?.runId,
              buildArtifactDownloadUrl: options?.buildArtifactDownloadUrl,
            });
          if (!fallbackArtifact) {
            return null;
          }
          const fileInsight = rowForValue(sourceFile || path);
          const figureKind = String(row.kind ?? "").trim().toLowerCase();
          const figureSummaryParts = [
            figureKind === "overlay_mip"
              ? "Maximum-intensity projection overlay"
              : figureKind === "overlay_mid_z"
                ? "Representative mid-Z overlay"
                : null,
            fileInsight?.coveragePercent !== null &&
            fileInsight?.coveragePercent !== undefined
              ? `coverage ${formatPercentMetric(fileInsight.coveragePercent)}`
              : toNumber(row.coverage_percent) !== null
                ? `coverage ${formatPercentMetric(toNumber(row.coverage_percent))}`
                : null,
            fileInsight?.objectCount !== null &&
            fileInsight?.objectCount !== undefined
              ? `${formatIntegerMetric(fileInsight.objectCount)} objects`
              : null,
            fileInsight?.activeSliceCount !== null &&
            fileInsight?.activeSliceCount !== undefined &&
            fileInsight?.zSliceCount !== null &&
            fileInsight?.zSliceCount !== undefined
              ? `${formatIntegerMetric(fileInsight.activeSliceCount)}/${formatIntegerMetric(fileInsight.zSliceCount)} active z-slices`
              : null,
          ].filter((value): value is string => value !== null);
          return {
            key: `${path}-${vizIndex}`,
            title: String(row.title ?? "Megaseg overlay").trim() || "Megaseg overlay",
            subtitle: sourceFile ? toDisplayFileLabel(sourceFile) : undefined,
            summary: figureSummaryParts.join(" · ") || undefined,
            previewUrl: fallbackArtifact.url,
            downloadUrl: fallbackArtifact.downloadUrl ?? fallbackArtifact.url,
            previewable: fallbackArtifact.previewable,
          } satisfies ScientificFigureCard;
        })
        .filter((row): row is ScientificFigureCard => row !== null)
        .sort((left, right) => {
          const leftKey = scientificFigureSortKey(left);
          const rightKey = scientificFigureSortKey(right);
          return leftKey[0] - rightKey[0] || leftKey[1].localeCompare(rightKey[1]);
        });

      const figureCardsFromHydratedArtifacts: ScientificFigureCard[] =
        figureCards.length > 0
          ? figureCards
          : runArtifacts
              .filter((artifact) =>
                /(megaseg.*overlay|overlay.*megaseg|overlay_mip|overlay_midz|mask_preview)/i.test(
                  `${artifact.path} ${artifact.title} ${artifact.sourceName}`
                )
              )
              .map((artifact, artifactIndex): ScientificFigureCard => ({
                key: `${artifact.path}-${artifactIndex}`,
                title: artifact.title || "Megaseg figure",
                subtitle:
                  artifact.sourcePath || artifact.sourceName
                    ? toDisplayFileLabel(artifact.sourcePath || artifact.sourceName)
                    : undefined,
                summary: /overlay_mip|mip/i.test(`${artifact.path} ${artifact.title}`)
                  ? "Maximum-intensity projection overlay"
                  : /overlay_midz|mid[-\s]?z/i.test(`${artifact.path} ${artifact.title}`)
                    ? "Representative mid-Z overlay"
                    : /mask_preview/i.test(`${artifact.path} ${artifact.title}`)
                      ? "Binary mask preview"
                      : undefined,
                previewUrl: artifact.url,
                downloadUrl: artifact.downloadUrl ?? artifact.url,
                previewable: artifact.previewable,
              }))
              .sort((left, right) => {
                const leftKey = scientificFigureSortKey(left);
                const rightKey = scientificFigureSortKey(right);
                return leftKey[0] - rightKey[0] || leftKey[1].localeCompare(rightKey[1]);
              });

      const megasegImages: ToolCardImage[] =
        figureCardsFromHydratedArtifacts.length > 0
          ? figureCardsFromHydratedArtifacts.flatMap((figure) => {
              const url = figure.previewUrl ?? figure.url;
              if (!url) {
                return [];
              }
              return [
                {
                  path: figure.key,
                  url,
                  downloadUrl: figure.downloadUrl ?? url,
                  title: figure.title,
                  sourceName: figure.file ?? figure.subtitle ?? figure.title,
                  previewable: figure.previewable,
                },
              ];
            })
          : fallbackMegasegImages.slice(0, 6);

      const hasMeaningfulMegasegResult =
        megasegSummary.success !== false &&
        (fileRows.length > 0 ||
          figureCardsFromHydratedArtifacts.length > 0 ||
          fallbackMegasegImages.length > 0 ||
          processed !== null);

      if (!hasMeaningfulMegasegResult) {
        return;
      }

      const firstRow = fileRows[0];
      const singleFile = fileRows.length <= 1;
      const metrics: ToolCardMetric[] = singleFile
        ? [
            {
              label: "Coverage",
              value: formatPercentMetric(firstRow?.coveragePercent ?? meanCoverage),
            },
            {
              label: "Objects",
              value: formatIntegerMetric(firstRow?.objectCount ?? meanObjectCount),
            },
            {
              label: "Active z-slices",
              value:
                firstRow?.activeSliceCount !== null &&
                firstRow?.activeSliceCount !== undefined &&
                firstRow?.zSliceCount !== null &&
                firstRow?.zSliceCount !== undefined
                  ? `${formatIntegerMetric(firstRow.activeSliceCount)}/${formatIntegerMetric(firstRow.zSliceCount)}`
                  : "n/a",
            },
            {
              label: "Largest component",
              value: formatIntegerMetric(firstRow?.largestComponentVoxels),
            },
          ]
        : [
            {
              label: "Processed",
              value:
                processed !== null && totalFiles !== null
                  ? `${formatIntegerMetric(processed)}/${formatIntegerMetric(totalFiles)}`
                  : formatIntegerMetric(processed),
            },
            {
              label: "Mean coverage",
              value: formatPercentMetric(meanCoverage),
            },
            {
              label: "Mean objects",
              value:
                meanObjectCount !== null ? meanObjectCount.toFixed(1) : "n/a",
            },
            {
              label: "Overlays",
              value: formatIntegerMetric(figureCards.length || fallbackMegasegImages.length),
            },
          ];

      cards.push({
        id: `${toolName}-${index}`,
        tool: "segment_image_megaseg",
        title: "Megaseg segmentation",
        subtitle:
          typeof megasegSummary.model === "string" && megasegSummary.model.trim()
            ? megasegSummary.model.trim()
            : undefined,
        metrics,
        classes: [],
        images: megasegImages,
        resourceRows: [],
        downloadRows: [],
        narrative: buildMegasegNarrative({
          fileRows,
          processed,
          meanCoverage,
          meanObjectCount,
        }),
        placement: "before_text",
        scientificFigures: figureCardsFromHydratedArtifacts,
        megasegInsights: {
          resultGroupId:
            String(megasegSummary.result_group_id ?? megasegSummary.output_directory ?? `${toolName}-${index}`),
          heroFigure:
            figureCardsFromHydratedArtifacts[0]
              ? {
                  key: figureCardsFromHydratedArtifacts[0].key,
                  kind: "overlay_mip",
                  title: figureCardsFromHydratedArtifacts[0].title,
                  file: figureCardsFromHydratedArtifacts[0].subtitle,
                  summary: figureCardsFromHydratedArtifacts[0].summary,
                  url: figureCardsFromHydratedArtifacts[0].previewUrl,
                  downloadUrl:
                    figureCardsFromHydratedArtifacts[0].downloadUrl ??
                    figureCardsFromHydratedArtifacts[0].previewUrl,
                  previewable: figureCardsFromHydratedArtifacts[0].previewable,
                }
              : null,
          secondaryFigures: figureCardsFromHydratedArtifacts.slice(1).map((figure) => ({
            key: figure.key,
            kind: /overlay_midz|mid[-\s]?z/i.test(`${figure.title} ${figure.summary}`)
              ? "overlay_mid_z"
              : /mask_preview/i.test(`${figure.title} ${figure.summary}`)
                ? "mask_preview"
                : "figure",
            title: figure.title,
            file: figure.subtitle,
            summary: figure.summary,
            url: figure.previewUrl,
            downloadUrl: figure.downloadUrl ?? figure.previewUrl,
            previewable: figure.previewable,
          })),
          fileRows,
          reportPath:
            typeof megasegSummary.report_path === "string" ? megasegSummary.report_path : null,
          summaryCsvPath:
            typeof megasegSummary.summary_csv_path === "string"
              ? megasegSummary.summary_csv_path
              : null,
          reportDownloadUrl: null,
          summaryCsvDownloadUrl: null,
          technicalSummary:
            typeof firstRow?.technicalSummary === "string" ? firstRow.technicalSummary : null,
          collectionLabel:
            processed !== null && totalFiles !== null
              ? `${formatIntegerMetric(processed)} of ${formatIntegerMetric(totalFiles)} images processed`
              : undefined,
          device:
            typeof megasegSummary.device === "string" && megasegSummary.device.trim()
              ? megasegSummary.device.trim()
              : null,
          structureChannel: toNumber(megasegSummary.structure_channel),
          nucleusChannel: toNumber(megasegSummary.nucleus_channel),
        },
      });
      return;
    }

    if (toolName === "estimate_depth_pro") {
      const summaryFiles = Array.isArray(summary?.files) ? summary.files : [];
      const processed = toNumber(summary?.processed);
      const totalFiles = toNumber(summary?.total_files);
      const meanDepth = toNumber(summary?.depth_mean_average);
      const depthRows = summaryFiles
        .slice(0, 8)
        .map((row) => toRecord(row))
        .filter((row): row is Record<string, unknown> => row !== null)
        .map((row) => ({
          rawFile: String(row.file ?? "file"),
          file: toDisplayFileLabel(String(row.file ?? "file")),
          depthMean: toNumber(row.depth_mean),
          depthMin: toNumber(row.depth_min),
          depthMax: toNumber(row.depth_max),
        }));

      const hoverDetailsByLookupKey = new Map<string, ToolImageHoverDetails>();
      depthRows.forEach((row) => {
        const details: ToolImageHoverDetails = {
          fileLabel:
            row.depthMean !== null && row.depthMin !== null && row.depthMax !== null
              ? `${row.file} • mean ${row.depthMean.toFixed(3)} • ${row.depthMin.toFixed(
                  3
                )}–${row.depthMax.toFixed(3)}`
              : row.file,
        };
        artifactLookupKeys(row.rawFile).forEach((key) => {
          if (!hoverDetailsByLookupKey.has(key)) {
            hoverDetailsByLookupKey.set(key, details);
          }
        });
      });

      if (matchedImages.length === 0) {
        runArtifacts
          .filter((artifact) =>
            /(depth|depth_map|overlay|side_by_side)/i.test(artifact.sourceName)
          )
          .slice(0, 6)
          .forEach((artifact) => addMatchedImage(artifact));
      }

      const matchedDepthImages: ToolCardImage[] = [...matchedImages]
        .sort((left, right) => Number(right.previewable) - Number(left.previewable))
        .slice(0, 6)
        .map((artifact) => {
          const details =
            artifactLookupKeys(artifact.sourceName)
              .map((key) => hoverDetailsByLookupKey.get(key))
              .find((value): value is ToolImageHoverDetails => value !== undefined) ??
            undefined;
          return {
            ...artifact,
            hoverDetails: details,
          };
        });

      cards.push({
        id: `${toolName}-${index}`,
        tool: "estimate_depth_pro",
        title: "DepthPro estimation",
        subtitle:
          typeof summary?.model === "string" && summary.model.length > 0
            ? summary.model
            : undefined,
        metrics: [
          {
            label: "Processed",
            value:
              processed !== null && totalFiles !== null
                ? `${processed}/${totalFiles}`
                : "n/a",
          },
          {
            label: "Mean depth",
            value: meanDepth !== null ? meanDepth.toFixed(4) : "n/a",
          },
        ],
        classes: [],
        images: matchedDepthImages,
        resourceRows: [],
        downloadRows: [],
      });
      return;
    }

    if (toolName === "quantify_segmentation_masks") {
      const summaryRecord = toRecord(summary?.summary) ?? {};
      const metricsMean = toRecord(summary?.metrics_mean) ?? {};
      const successfulMasks = toNumber(summaryRecord?.successful_masks);
      const meanCoverage = toNumber(summaryRecord?.mean_coverage_percent);
      const meanObjectCount = toNumber(summaryRecord?.mean_object_count);
      const rowCount = toNumber(summary?.row_count);
      const diceMean = toNumber(metricsMean?.dice);
      const hasMeaningfulQuantification =
        successfulMasks !== null ||
        meanCoverage !== null ||
        meanObjectCount !== null ||
        rowCount !== null ||
        diceMean !== null;

      if (!hasMeaningfulQuantification) {
        return;
      }

      cards.push({
        id: `${toolName}-${index}`,
        tool: "quantify_segmentation_masks",
        title: "Segmentation quantification",
        subtitle:
          typeof summary?.evaluation_pairing_fallback_used === "boolean" &&
          summary.evaluation_pairing_fallback_used
            ? "evaluation paired by index fallback"
            : undefined,
        metrics: [
          {
            label: "Masks",
            value: successfulMasks !== null ? `${Math.round(successfulMasks)}` : "n/a",
          },
          {
            label: "Mean coverage",
            value: meanCoverage !== null ? `${meanCoverage.toFixed(2)}%` : "n/a",
          },
          {
            label: "Mean objects",
            value: meanObjectCount !== null ? `${meanObjectCount.toFixed(2)}` : "n/a",
          },
          ...(rowCount !== null ? [{ label: "Rows", value: `${Math.round(rowCount)}` }] : []),
          ...(diceMean !== null ? [{ label: "Mean Dice", value: diceMean.toFixed(3) }] : []),
        ],
        classes: [],
        images: [],
        resourceRows: [],
        downloadRows: [],
      });
      return;
    }

    if (toolName === "plot_quantified_detections") {
      if (matchedImages.length === 0) {
        runArtifacts
          .filter((artifact) => isPlotArtifact(artifact))
          .slice(0, 6)
          .forEach((artifact) => addMatchedImage(artifact));
      }
      const plottedImages = matchedImages.some((artifact) => isPlotArtifact(artifact))
        ? matchedImages.filter((artifact) => isPlotArtifact(artifact))
        : [...matchedImages];
      const summaryRecord = toRecord(summary?.summary) ?? summary;
      const totalDetections = toNumber(summaryRecord?.total_detections);
      const lowConfidenceCount = toNumber(summaryRecord?.low_confidence_count);
      const confidenceThreshold = toNumber(summaryRecord?.confidence_threshold);
      const generatedPlotCount = toNumber(summaryRecord?.generated_plot_count);
      cards.push({
        id: `${toolName}-${index}`,
        tool: "plot_quantified_detections",
        title: "Detection uncertainty plots",
        subtitle:
          typeof summaryRecord?.image_name === "string" && summaryRecord.image_name.trim()
            ? summaryRecord.image_name.trim()
            : undefined,
        metrics: [
          {
            label: "Detections",
            value: totalDetections !== null ? `${Math.round(totalDetections)}` : "n/a",
          },
          {
            label: "Below threshold",
            value: lowConfidenceCount !== null ? `${Math.round(lowConfidenceCount)}` : "n/a",
          },
          {
            label: "Threshold",
            value:
              confidenceThreshold !== null ? `${(confidenceThreshold * 100).toFixed(0)}%` : "n/a",
          },
          {
            label: "Plots",
            value: generatedPlotCount !== null ? `${Math.round(generatedPlotCount)}` : `${plottedImages.length}`,
          },
        ],
        classes: [],
        images: plottedImages,
        resourceRows: [],
        downloadRows: [],
      });
      return;
    }

    const summaryClasses = Array.isArray(summary?.classes) ? summary.classes : [];
    const summaryDetections = Array.isArray(summary?.detections)
      ? summary.detections
      : [];
    const summaryPredictions = Array.isArray(summary?.predictions)
      ? summary.predictions
      : [];
    const summaryMetrics = toRecord(summary?.metrics);
    const scientificSummary = toRecord(summary?.scientific_summary);
    const scientificOverall = toRecord(scientificSummary?.overall);
    const inferenceConfiguration =
      toRecord(summary?.inference_configuration) ?? toRecord(scientificSummary?.inference);
    const spatialAnalysis = toRecord(summary?.spatial_analysis);
    const overallPrairieBurrowContext = toRecord(
      spatialAnalysis?.overall_prairie_burrow_context
    );
    const metadataSummary =
      toRecord(spatialAnalysis?.metadata_summary) ?? toRecord(scientificSummary?.metadata);
    const predictionImageRecords = Array.isArray(summary?.prediction_image_records)
      ? summary.prediction_image_records
      : Array.isArray(scientificSummary?.image_records)
        ? scientificSummary.image_records
        : [];
    const predictionImagePaths = Array.isArray(summary?.prediction_images)
      ? summary.prediction_images
      : [];
    const predictionImageRawPaths = Array.isArray(summary?.prediction_images_raw)
      ? summary.prediction_images_raw
      : [];
    const spatialImagesRaw = Array.isArray(spatialAnalysis?.images)
      ? spatialAnalysis.images
      : Array.isArray(scientificSummary?.per_image)
        ? scientificSummary.per_image
        : [];
    const prairieImageAnalyses = spatialImagesRaw
      .map((item) => toPrairieImageAnalysis(item))
      .filter((item): item is PrairieImageAnalysis => item !== null);
    const predictionDetectionRows = summaryPredictions.flatMap((item) => {
      const row = toRecord(item);
      if (!row) {
        return [];
      }
      const rawFile = String(row.path ?? row.input_path ?? "").trim();
      return (Array.isArray(row.boxes) ? row.boxes : [])
        .map((box) => {
          const parsed = toRecord(box);
          if (!parsed) {
            return null;
          }
          return {
            ...parsed,
            file: rawFile || parsed.file,
          };
        })
        .filter((entry) => entry !== null) as Record<string, unknown>[];
    });
    const detectionRows =
      summaryDetections.length > 0 ? summaryDetections : predictionDetectionRows;

    const detectionsByFile = new Map<
      string,
      { fileLabel?: string; detectionBoxes: ToolDetectionBox[] }
    >();
    detectionRows.forEach((item) => {
      const row = toRecord(item);
      if (!row) {
        return;
      }
      const parsedBox = toToolDetectionBox(row);
      if (!parsedBox) {
        return;
      }
      const rawFile = String(row.file ?? "").trim();
      const key = rawFile || "__global__";
      const existing = detectionsByFile.get(key) ?? {
        fileLabel: rawFile ? toDisplayFileLabel(rawFile) : undefined,
        detectionBoxes: [],
      };
      if (existing.detectionBoxes.length < 120) {
        existing.detectionBoxes.push(parsedBox);
      }
      detectionsByFile.set(key, existing);
    });

    const prairieAnalysisByLookupKey = new Map<string, PrairieImageAnalysis>();
    prairieImageAnalyses.forEach((analysis) => {
      artifactLookupKeys(analysis.rawFile).forEach((lookupKey) => {
        if (!prairieAnalysisByLookupKey.has(lookupKey)) {
          prairieAnalysisByLookupKey.set(lookupKey, analysis);
        }
      });
    });

    const rawDetectionFiles = Array.from(
      new Set([
        ...prairieImageAnalyses.map((item) => item.rawFile),
        ...Array.from(detectionsByFile.keys()).filter((rawFile) => rawFile !== "__global__"),
      ])
    );
    rawDetectionFiles.forEach((rawFile) => {
      artifactLookupKeys(rawFile).forEach((key) => {
        const matches = artifactBySource.get(key) ?? [];
        matches.forEach((match) => addMatchedImage(match));
      });
    });
    if (matchedImages.length === 0 && rawDetectionFiles.length > 0) {
      const prairieFallbackImages = runArtifacts
        .filter((artifact) => isImageArtifactPath(artifact.path))
        .filter((artifact) => {
          const haystack = `${artifact.sourceName} ${artifact.path}`.toLowerCase();
          return !/(predict|prediction|yolo|det-|overlay|mask_preview|side_by_side|labeled)/i.test(
            haystack
          );
        })
        .slice(0, 6);
      prairieFallbackImages.forEach((artifact) => addMatchedImage(artifact));
    }
    if (matchedImages.length === 0 && rawDetectionFiles.length > 0) {
      toolCardImagesFromUploadedMatches(
        rawDetectionFiles,
        uploadedPreviewLookup,
        buildUploadPreviewUrl
      ).forEach((image) => addMatchedImage(image));
    }

    const yoloHoverDetailsByLookupKey = new Map<string, ToolImageHoverDetails>();
    detectionsByFile.forEach((details, rawFile) => {
      if (rawFile === "__global__") {
        return;
      }
      const payload: ToolImageHoverDetails = {
        fileLabel: details.fileLabel,
        detectionBoxes: details.detectionBoxes,
      };
      artifactLookupKeys(rawFile).forEach((lookupKey) => {
        if (!yoloHoverDetailsByLookupKey.has(lookupKey)) {
          yoloHoverDetailsByLookupKey.set(lookupKey, payload);
        }
      });
    });
    const fallbackPrairieAnalysis =
      prairieImageAnalyses.length === 1 ? prairieImageAnalyses[0] : undefined;
    const fallbackYoloDetails = (() => {
      const entries = Array.from(detectionsByFile.entries());
      const nonGlobal = entries.filter(([rawFile]) => rawFile !== "__global__");
      if (nonGlobal.length === 1) {
        const [rawFile, details] = nonGlobal[0];
        return {
          fileLabel: details.fileLabel ?? toDisplayFileLabel(rawFile),
          detectionBoxes: details.detectionBoxes,
          prairieImageAnalysis:
            fallbackPrairieAnalysis && fallbackPrairieAnalysis.rawFile === rawFile
              ? fallbackPrairieAnalysis
              : undefined,
        } satisfies ToolImageHoverDetails;
      }
      if (
        (nonGlobal.length === 0 || nonGlobal.length === 1) &&
        detectionsByFile.has("__global__")
      ) {
        const globalDetails = detectionsByFile.get("__global__");
        if (globalDetails) {
          return {
            fileLabel: "Detected objects",
            detectionBoxes: globalDetails.detectionBoxes,
            prairieImageAnalysis: fallbackPrairieAnalysis,
          } satisfies ToolImageHoverDetails;
        }
      }
      if (fallbackPrairieAnalysis) {
        return {
          fileLabel: fallbackPrairieAnalysis.fileLabel,
          prairieImageAnalysis: fallbackPrairieAnalysis,
        } satisfies ToolImageHoverDetails;
      }
      return undefined;
    })();

    if (matchedImages.length === 0) {
      runArtifacts
        .filter((artifact) => /(det-|predict|yolo)/i.test(artifact.sourceName))
        .slice(0, 4)
        .forEach((artifact) => addMatchedImage(artifact));
    }

    const classes = summaryClasses
      .map((item) => toRecord(item))
      .filter((item): item is Record<string, unknown> => item !== null)
      .map((item) => ({
        name: String(item.class_name ?? item.name ?? "class"),
        count: Math.max(0, Math.round(toNumber(item.count) ?? 0)),
      }))
      .filter((item) => item.count > 0)
      .slice(0, 8);

    const hasPrairieAnalysisSignal = prairieImageAnalyses.some(
      (analysis) =>
        analysis.prairieDogCount !== null && analysis.prairieDogCount !== undefined ||
        analysis.burrowCount !== null && analysis.burrowCount !== undefined ||
        analysis.nearestBurrowDistancePxMean !== null &&
          analysis.nearestBurrowDistancePxMean !== undefined ||
        analysis.nearestBurrowDistancePxMin !== null &&
          analysis.nearestBurrowDistancePxMin !== undefined ||
        analysis.overlappingBurrowCount !== null &&
          analysis.overlappingBurrowCount !== undefined
    );

    const isPrairieDetection =
      classes.some((item) => isPrairieDetectionClassName(item.name)) ||
      hasPrairieAnalysisSignal ||
      toNumber(summaryMetrics?.prairie_dog_count) !== null ||
      toNumber(summaryMetrics?.burrow_count) !== null;
    const avgConfidence =
      toNumber(summaryMetrics?.avg_confidence) ?? toNumber(summary?.avg_confidence);
    const prairieDogCount = Math.max(
      0,
      Math.round(
        toNumber(summaryMetrics?.prairie_dog_count) ??
          toNumber(scientificOverall?.prairie_dog_count) ??
          classes.find((item) => item.name === "prairie_dog")?.count ??
          0
      )
    );
    const burrowCount = Math.max(
      0,
      Math.round(
        toNumber(summaryMetrics?.burrow_count) ??
          toNumber(scientificOverall?.burrow_count) ??
          classes.find((item) => item.name === "burrow")?.count ??
          0
      )
    );
    const nearestBurrowDistancePxMean =
      toNumber(overallPrairieBurrowContext?.nearest_burrow_distance_px_mean) ??
      toNumber(scientificOverall?.nearest_burrow_distance_px_mean);
    const matchedYoloImages = [...matchedImages]
      .sort((left, right) => Number(right.previewable) - Number(left.previewable))
      .slice(0, 6)
      .map((artifact) => {
        const lookupKeys = artifactLookupKeys(artifact.sourceName);
        const detectionDetails =
          lookupKeys
            .map((key) => yoloHoverDetailsByLookupKey.get(key))
            .find((value): value is ToolImageHoverDetails => value !== undefined) ??
          undefined;
        const prairieAnalysis =
          lookupKeys
            .map((key) => prairieAnalysisByLookupKey.get(key))
            .find((value): value is PrairieImageAnalysis => value !== undefined) ??
          detectionDetails?.prairieImageAnalysis ??
          fallbackYoloDetails?.prairieImageAnalysis;
        return {
          ...artifact,
          hoverDetails:
            detectionDetails || prairieAnalysis || fallbackYoloDetails
              ? {
                  ...(detectionDetails ?? fallbackYoloDetails ?? {}),
                  prairieImageAnalysis: prairieAnalysis,
                }
              : undefined,
        };
      });
    const yoloFiguresFromRecords = buildYoloFigureCards(
      predictionImageRecords,
      artifactBySource,
      uploadedPreviewLookup,
      buildUploadPreviewUrl
    );
    const yoloFiguresFromPaths: YoloFigureCard[] = predictionImagePaths
      .map((previewPath, figureIndex): YoloFigureCard | null => {
        const rawSourcePath = String(predictionImageRawPaths[figureIndex] ?? "").trim();
        const resolvedRawSourcePath = rawSourcePath || previewPath;
        const allowsOriginalDisplayFallback =
          previewPath.length === 0 || previewPath === resolvedRawSourcePath;
      const previewArtifact =
        resolveArtifactForLookup(previewPath, artifactBySource) ??
        resolveArtifactForLookup(extractFilename(previewPath), artifactBySource) ??
        (allowsOriginalDisplayFallback
          ? resolveArtifactForLookup(resolvedRawSourcePath, artifactBySource) ??
            resolveArtifactForLookup(extractFilename(resolvedRawSourcePath), artifactBySource) ??
            uploadedPreviewArtifactFromPath(
              previewPath || resolvedRawSourcePath,
              uploadedPreviewLookup,
              buildUploadPreviewUrl
            )
          : undefined);
      const rawArtifact =
        resolveArtifactForLookup(resolvedRawSourcePath, artifactBySource) ??
        resolveArtifactForLookup(extractFilename(resolvedRawSourcePath), artifactBySource) ??
        uploadedPreviewArtifactFromPath(
          resolvedRawSourcePath,
          uploadedPreviewLookup,
          buildUploadPreviewUrl
        );
        const displayedArtifact = previewArtifact ?? (allowsOriginalDisplayFallback ? rawArtifact : undefined);
        if (!displayedArtifact) {
          return null;
        }
        const title =
          toDisplayFileLabel(
            extractFilename(resolvedRawSourcePath) ||
              extractFilename(previewPath) ||
              displayedArtifact.sourceName
          ) ||
          artifactTitleFromPath(previewPath || resolvedRawSourcePath || displayedArtifact.path) ||
          `Detection ${figureIndex + 1}`;
        const figure: YoloFigureCard = {
          key: `${resolvedRawSourcePath || previewPath || displayedArtifact.path}-${figureIndex}`,
          title,
          subtitle: previewArtifact ? "" : "Original image",
          previewUrl: displayedArtifact.url,
          downloadUrl: displayedArtifact.downloadUrl ?? displayedArtifact.url,
          originalUrl: rawArtifact?.downloadUrl ?? rawArtifact?.url,
          sourcePath: previewPath || undefined,
          rawSourcePath: resolvedRawSourcePath || undefined,
          sourceName: extractFilename(previewPath) || undefined,
          rawSourceName: extractFilename(resolvedRawSourcePath) || undefined,
          boxCount: null,
          classCounts: [],
          previewKind: previewArtifact ? "matplotlib_annotated" : "original_fallback",
          previewable: displayedArtifact.previewable,
        };
        return figure;
      })
      .filter((item): item is YoloFigureCard => item !== null);
    const yoloFiguresFromArtifacts = buildYoloFigureCardsFromAnnotatedArtifacts(
      runArtifacts.filter((artifact) => isMatplotlibAnnotatedArtifact(artifact)),
      predictionImageRecords,
      predictionImageRawPaths.map((value) => String(value || "").trim()),
      prairieImageAnalyses,
      artifactBySource,
      uploadedPreviewLookup,
      buildUploadPreviewUrl
    );
    const yoloFigures =
      yoloFiguresFromRecords.length > 0
        ? yoloFiguresFromRecords
        : yoloFiguresFromPaths.length > 0
          ? yoloFiguresFromPaths
          : yoloFiguresFromArtifacts;
    const missingAnnotatedFigure =
      (predictionImageRecords.length > 0 ||
        predictionImagePaths.length > 0 ||
        matchedYoloImages.length > 0) &&
      yoloFigures.length === 0;
    const yoloCardImages = yoloFigures.map((figure) => ({
      path: figure.sourcePath ?? figure.rawSourcePath ?? figure.previewUrl,
      url: figure.previewUrl,
      title: figure.title,
      sourceName:
        figure.sourceName ?? figure.rawSourceName ?? figure.sourcePath ?? figure.previewUrl,
      previewable: figure.previewable,
      downloadUrl: figure.downloadUrl ?? figure.previewUrl,
    }));

    if (isPrairieDetection) {
      const analysisSummary =
        typeof summary?.analysis_summary === "string" && summary.analysis_summary.trim()
          ? summary.analysis_summary.trim()
          : undefined;
      const metadataInsights = {
        capturedAt: String(metadataSummary?.first_captured_at ?? "").trim() || undefined,
        latitude: toNumber(metadataSummary?.first_latitude),
        longitude: toNumber(metadataSummary?.first_longitude),
      };
      cards.push({
        id: `${toolName}-${index}`,
        tool: "yolo_detect",
        title: "Prairie dog survey",
        subtitle: typeof summary?.model_name === "string" ? summary.model_name : undefined,
        metrics: [
          {
            label: "Prairie dogs",
            value: `${prairieDogCount}`,
          },
          {
            label: "Burrows",
            value: `${burrowCount}`,
          },
          {
            label: "Avg confidence",
            value: avgConfidence !== null ? `${(avgConfidence * 100).toFixed(1)}%` : "n/a",
          },
          {
            label: "Nearest burrow",
            value:
              nearestBurrowDistancePxMean !== null
                ? `${nearestBurrowDistancePxMean.toFixed(1)} px`
                : burrowCount > 0
                  ? "n/a"
                  : "none detected",
          },
        ],
        classes,
        images: yoloCardImages,
        resourceRows: [],
        downloadRows: [],
        variant: "prairie_detection",
        narrative: analysisSummary,
        yoloFigures,
        yoloFigureAvailability: { missingAnnotatedFigure },
        prairieInsights: {
          summary: analysisSummary,
          inferenceBackend:
            typeof inferenceConfiguration?.backend === "string"
              ? inferenceConfiguration.backend
              : undefined,
          tileSize: toNumber(inferenceConfiguration?.tile_size),
          tileOverlap: toNumber(inferenceConfiguration?.tile_overlap),
          tileCount: toNumber(inferenceConfiguration?.tile_count),
          conf: toNumber(inferenceConfiguration?.conf),
          iou: toNumber(inferenceConfiguration?.iou),
          mergeIou: toNumber(inferenceConfiguration?.merge_iou),
          prairieDogCount,
          burrowCount,
          avgConfidence,
          nearestBurrowDistancePxMean,
          nearestBurrowDistancePxMin: toNumber(
            overallPrairieBurrowContext?.nearest_burrow_distance_px_min
          ),
          overlapCount: toNumber(
            overallPrairieBurrowContext?.prairie_dogs_overlapping_burrows
          ),
          metadataSummary: metadataInsights,
          perImage: prairieImageAnalyses,
        },
      });
      return;
    }

    cards.push({
      id: `${toolName}-${index}`,
      tool: "yolo_detect",
      title: "YOLO detection",
      subtitle: typeof summary?.model_name === "string" ? summary.model_name : undefined,
      metrics: [
        {
          label: "Total boxes",
          value:
            toNumber(summary?.total_boxes) !== null
              ? `${Math.round(toNumber(summary?.total_boxes) ?? 0)}`
              : "0",
        },
        {
          label: "Avg confidence",
          value: avgConfidence !== null ? `${(avgConfidence * 100).toFixed(1)}%` : "n/a",
        },
        {
          label: "Finetune",
          value: summary?.finetune_recommended ? "recommended" : "optional",
        },
      ],
      classes,
      images: yoloCardImages,
      resourceRows: [],
      downloadRows: [],
      yoloFigures,
      yoloFigureAvailability: { missingAnnotatedFigure },
    });
  });

  let mergedBisqueSearchCard: ToolResultCard | null = null;
  let mergedBisqueSearchMatchCount = 0;

  if (bisqueSearchByType.size > 0 && !hasSuccessfulBisqueModule) {
    const selectedBisqueSearches = Array.from(bisqueSearchByType.values()).sort(
      (left, right) => left.index - right.index
    );
    const uniqueRows = new Map<string, ToolResourceRow>();
    selectedBisqueSearches.forEach((candidate) => {
      candidate.resourceRows.forEach((row) => {
        const key =
          row.uri?.toLowerCase() ||
          `${row.name.toLowerCase()}|${String(row.owner ?? "").toLowerCase()}|${String(
            row.created ?? ""
          ).toLowerCase()}|${String(row.resourceType ?? "").toLowerCase()}`;
        if (!uniqueRows.has(key)) {
          uniqueRows.set(key, row);
        }
      });
    });
    const mergedRows = Array.from(uniqueRows.values()).slice(0, 12);
    const totalMatches = selectedBisqueSearches.reduce((sum, candidate) => {
      const countValue = candidate.matchCount ?? candidate.resourceRows.length;
      return sum + Math.max(0, Math.round(countValue));
    }, 0);
    if (totalMatches <= 0 && mergedRows.length === 0) {
      mergedBisqueSearchMatchCount = 0;
      mergedBisqueSearchCard = null;
    } else {
      const resourceTypes = Array.from(
        new Set(
          selectedBisqueSearches
            .map((candidate) => String(candidate.resourceType ?? "").trim().toLowerCase())
            .filter((value) => value.length > 0)
        )
      );
      const subtitle =
        resourceTypes.length === 0
          ? undefined
          : resourceTypes.length === 1
            ? `${resourceTypes[0]} resources`
            : `${resourceTypes.slice(0, 2).join(" + ")}${resourceTypes.length > 2 ? " + more" : ""} resources`;
      const lastCandidate = selectedBisqueSearches[selectedBisqueSearches.length - 1];
      mergedBisqueSearchMatchCount = totalMatches;
      mergedBisqueSearchCard = {
        id: `bisque-search-${lastCandidate.index}`,
        tool:
          selectedBisqueSearches.length === 1 &&
          selectedBisqueSearches[0].toolName === "bisque_advanced_search"
            ? "bisque_advanced_search"
            : "search_bisque_resources",
        title:
          selectedBisqueSearches.length === 1 &&
          selectedBisqueSearches[0].toolName === "bisque_advanced_search"
            ? "BisQue advanced search"
            : "BisQue search",
        subtitle,
        metrics: [
          {
            label: "Matches",
            value: `${totalMatches}`,
          },
          {
            label: "Shown",
            value: `${mergedRows.length}`,
          },
        ],
        classes: [],
        images: toolCardImagesFromBisqueResourceRows(mergedRows, 1),
        resourceRows: mergedRows,
        downloadRows: [],
      };
    }
  }

  let primaryBisqueCard: ToolResultCard | null = null;
  if (latestBisqueUploadCard) {
    primaryBisqueCard = latestBisqueUploadCard;
  }

  if (!primaryBisqueCard && bestBisqueFindAssetsCard && !hasSuccessfulBisqueModule) {
    const findAssetsCard = bestBisqueFindAssetsCard as ToolResultCard;
    if (mergedBisqueSearchCard) {
      const mergedRows = mergeBisqueResourceRows(
        findAssetsCard.resourceRows,
        mergedBisqueSearchCard.resourceRows
      );
      const existingMetrics = new Map(
        findAssetsCard.metrics.map((metric) => [metric.label, metric.value] as const)
      );
      const existingMatchCount = Number.parseInt(existingMetrics.get("Matches") ?? "0", 10);
      const mergedMatchCount = Math.max(
        Number.isFinite(existingMatchCount) ? existingMatchCount : 0,
        mergedBisqueSearchMatchCount,
        mergedRows.length
      );
      primaryBisqueCard = {
        ...findAssetsCard,
        subtitle: findAssetsCard.subtitle ?? mergedBisqueSearchCard.subtitle,
        metrics: findAssetsCard.metrics.map((metric) => {
          if (metric.label === "Matches") {
            return { ...metric, value: `${mergedMatchCount}` };
          }
          if (metric.label === "Shown") {
            return { ...metric, value: `${mergedRows.length}` };
          }
          return metric;
        }),
        images:
          findAssetsCard.images.length > 0
            ? findAssetsCard.images
            : toolCardImagesFromBisqueResourceRows(mergedRows, 1),
        resourceRows: mergedRows,
      };
    } else {
      primaryBisqueCard = findAssetsCard;
    }
  } else if (mergedBisqueSearchCard && !primaryBisqueCard) {
    primaryBisqueCard = mergedBisqueSearchCard;
  }

  if (!primaryBisqueCard && bisqueMetadataByResource.size > 0) {
    primaryBisqueCard = Array.from(bisqueMetadataByResource.values())
      .sort((left, right) => {
        if (left.infoScore !== right.infoScore) {
          return right.infoScore - left.infoScore;
        }
        return right.index - left.index;
      })[0]?.card ?? null;
  }

  if (primaryBisqueCard) {
    cards.push(primaryBisqueCard);
  } else if (bisqueDownloadRows.length > 0) {
    cards.push({
      id: `bisque-download-${latestBisqueDownloadIndex >= 0 ? latestBisqueDownloadIndex : cards.length}`,
      tool: "bisque_download_resource",
      title: "BisQue downloads",
      metrics: [
        {
          label: "Files",
          value: `${bisqueDownloadRows.length}`,
        },
        {
          label: "Succeeded",
          value: `${bisqueDownloadRows.filter((row) => row.status === "ok").length}/${bisqueDownloadRows.length}`,
        },
      ],
      classes: [],
      images: [],
      resourceRows: [],
      downloadRows: bisqueDownloadRows.slice(0, 12),
    });
  }

  return cards;
};

const mergeBisqueResourceRows = (rows: ToolResourceRow[]): ToolResourceRow[] => {
  const merged = new Map<string, ToolResourceRow>();
  rows.forEach((row) => {
    const key =
      row.resourceUri?.toLowerCase() ||
      row.clientViewUrl?.toLowerCase() ||
      row.uri?.toLowerCase() ||
      row.name.toLowerCase();
    if (!merged.has(key)) {
      merged.set(key, row);
    }
  });
  return Array.from(merged.values());
};

const extractSearchResourceRowsFromMessage = (message: UiMessage): ToolResourceRow[] => {
  const cards = buildToolResultCards(message.progressEvents ?? [], message.runArtifacts ?? []);
  return mergeBisqueResourceRows(
    cards
      .filter((card) =>
        card.tool === "search_bisque_resources" ||
        card.tool === "bisque_advanced_search" ||
        card.tool === "bisque_find_assets"
      )
      .flatMap((card) => card.resourceRows)
      .filter((row) => Boolean(row.resourceUri))
  );
};

const extractResolvedBisqueRowsFromMessage = (message: UiMessage): ToolResourceRow[] =>
  mergeBisqueResourceRows(message.resolvedBisqueResources ?? []);

const extractArtifactHandlesFromResponseMetadata = (
  responseMetadata: Record<string, unknown> | null | undefined
): Record<string, string[]> => {
  const metadata = toRecord(responseMetadata);
  const researchProgram = toRecord(metadata?.research_program);
  const proMode = toRecord(metadata?.pro_mode);
  const proModeResearchProgram = toRecord(proMode?.research_program);
  return toArtifactHandleMap(
    researchProgram?.handles ?? proModeResearchProgram?.handles ?? {}
  );
};

const mergeSelectionContexts = (
  primary: SelectionContext | null,
  secondary: SelectionContext | null
): SelectionContext | null => {
  if (!primary) {
    return secondary;
  }
  if (!secondary) {
    return primary;
  }
  return {
    context_id: primary.context_id ?? secondary.context_id ?? makeId(),
    source: primary.source ?? secondary.source ?? null,
    focused_file_ids: uniqueFileIds([
      ...(primary.focused_file_ids ?? []),
      ...(secondary.focused_file_ids ?? []),
    ]),
    resource_uris: Array.from(
      new Set([...(primary.resource_uris ?? []), ...(secondary.resource_uris ?? [])])
    ),
    dataset_uris: Array.from(
      new Set([...(primary.dataset_uris ?? []), ...(secondary.dataset_uris ?? [])])
    ),
    artifact_handles: {
      ...toArtifactHandleMap(secondary.artifact_handles),
      ...toArtifactHandleMap(primary.artifact_handles),
    },
    originating_message_id:
      primary.originating_message_id ?? secondary.originating_message_id ?? null,
    originating_user_text:
      primary.originating_user_text ?? secondary.originating_user_text ?? null,
    suggested_domain: primary.suggested_domain ?? secondary.suggested_domain ?? null,
    suggested_tool_names: Array.from(
      new Set([...(primary.suggested_tool_names ?? []), ...(secondary.suggested_tool_names ?? [])])
    ),
  };
};

const deriveBisqueSelectionContextFromResponseMetadata = ({
  responseMetadata,
  source,
  originatingUserText,
  suggestedDomain,
}: {
  responseMetadata: Record<string, unknown> | null | undefined;
  source: SelectionContext["source"];
  originatingUserText: string;
  suggestedDomain?: SelectionContext["suggested_domain"];
}): {
  selectionContext: SelectionContext | null;
  resolvedRows: ToolResourceRow[];
  clearsSelection: boolean;
} => {
  const metadata = toRecord(responseMetadata);
  const artifactHandles = extractArtifactHandlesFromResponseMetadata(responseMetadata);
  const toolInvocations = Array.isArray(metadata?.tool_invocations)
    ? metadata.tool_invocations
    : [];
  const clearsSelection = toolInvocations.some((entry) => {
    const record = toRecord(entry);
    return String(record?.tool ?? "").trim() === "delete_bisque_resource";
  });
  const resolvedRows = mergeBisqueResourceRows(
    toolInvocations.flatMap((entry) => {
      const record = toRecord(entry);
      if (!record) {
        return [];
      }
      const summaryRows = toToolResourceRows((record.output_summary as Record<string, unknown> | undefined)?.rows);
      if (summaryRows.length > 0) {
        return summaryRows;
      }

      const envelope = toRecord(record.output_envelope);
      if (!envelope) {
        return [];
      }

      const rowsFromEnvelope = toToolResourceRows(envelope.rows);
      if (rowsFromEnvelope.length > 0) {
        return rowsFromEnvelope;
      }

      const resultRows = Array.isArray(envelope.results)
        ? envelope.results
            .map((item) => toRecord(item))
            .filter((item): item is Record<string, unknown> => item !== null)
            .map((item) => ({
              name:
                String(item.file ?? item.name ?? "").trim() ||
                String(record.tool ?? "resource").trim() ||
                "resource",
              resource_uri: String(item.resource_uri ?? "").trim() || undefined,
              client_view_url: String(item.client_view_url ?? "").trim() || undefined,
              image_service_url: String(item.image_service_url ?? "").trim() || undefined,
              uri:
                String(item.client_view_url ?? item.resource_uri ?? "").trim() || undefined,
            }))
        : [];
      if (resultRows.length > 0) {
        return toToolResourceRows(resultRows);
      }

      const resourceUri = String(
        envelope.resource_uri ?? envelope.deleted_uri ?? envelope.uri ?? ""
      ).trim();
      const clientViewUrl = String(
        envelope.client_view_url ?? envelope.deleted_client_view_url ?? ""
      ).trim() || toBisqueClientViewUrl(resourceUri) || "";
      const datasetUri = String(envelope.dataset_uri ?? "").trim();
      const datasetClientViewUrl =
        String(envelope.dataset_client_view_url ?? "").trim() || toBisqueClientViewUrl(datasetUri) || "";
      const directRows: Array<Record<string, unknown>> = [];
      if (resourceUri) {
        directRows.push({
          name:
            String(envelope.resource_name ?? envelope.name ?? "").trim() ||
            String(record.tool ?? "resource").trim() ||
          "resource",
          resource_uri: resourceUri,
          client_view_url: clientViewUrl || undefined,
          uri: clientViewUrl || resourceUri,
        });
      }
      if (datasetUri) {
        directRows.push({
          name:
            String(envelope.dataset_name ?? "").trim() ||
            "dataset",
          resource_uri: datasetUri,
          client_view_url: datasetClientViewUrl || undefined,
          uri: datasetClientViewUrl || datasetUri,
          resource_type: "dataset",
        });
      }
      return toToolResourceRows(directRows);
    })
  );

  if (resolvedRows.length === 0) {
    if (Object.keys(artifactHandles).length > 0) {
      return {
        selectionContext: buildBisqueSelectionContext({
          source,
          artifactHandles,
          originatingUserText,
          suggestedDomain,
          suggestedToolNames: [],
        }),
        resolvedRows,
        clearsSelection,
      };
    }
    return {
      selectionContext: null,
      resolvedRows,
      clearsSelection,
    };
  }

  const partitioned = partitionBisqueRowsByUri(resolvedRows);
  return {
    selectionContext: buildBisqueSelectionContext({
      source,
      resourceUris: partitioned.resourceUris,
      datasetUris: partitioned.datasetUris,
      artifactHandles,
      originatingUserText,
      suggestedDomain,
      suggestedToolNames: [],
    }),
    resolvedRows,
    clearsSelection,
  };
};

const bisqueNumberWords: Record<string, number> = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
};

const parseBisqueSelectionCount = (promptText: string): number | null => {
  const lowered = String(promptText || "").trim().toLowerCase();
  const firstMatch = lowered.match(
    /\bfirst\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b/
  );
  if (firstMatch?.[1]) {
    const raw = firstMatch[1];
    const parsed = Number(raw);
    if (Number.isFinite(parsed) && parsed > 0) {
      return Math.min(12, Math.floor(parsed));
    }
    return bisqueNumberWords[raw] ?? null;
  }
  if (/\b(first one|the first one|the first image|the first file)\b/.test(lowered)) {
    return 1;
  }
  const quantityMatch = lowered.match(
    /\b(?:correct|these|those|them|all|show|preview|view|open|make|create|build|use)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:image|images|file|files|resource|resources)\b/
  );
  if (quantityMatch?.[1]) {
    const raw = quantityMatch[1];
    const parsed = Number(raw);
    if (Number.isFinite(parsed) && parsed > 0) {
      return Math.min(12, Math.floor(parsed));
    }
    return bisqueNumberWords[raw] ?? null;
  }
  return null;
};

const bisquePromptTypeHint = (
  promptText: string
): "tiff" | "png" | "table" | "dataset" | null => {
  const lowered = String(promptText || "").trim().toLowerCase();
  if (/\b(?:ome[-\s]?tiff?|tiff?|tif)\b/.test(lowered)) {
    return "tiff";
  }
  if (/\bpng\b/.test(lowered)) {
    return "png";
  }
  if (/\b(?:hdf5|h5|table|tables|dream3d)\b/.test(lowered)) {
    return "table";
  }
  if (/\bdataset\b/.test(lowered)) {
    return "dataset";
  }
  return null;
};

const bisqueRowMatchesTypeHint = (
  row: ToolResourceRow,
  typeHint: "tiff" | "png" | "table" | "dataset" | null
): boolean => {
  if (!typeHint) {
    return true;
  }
  const haystack = `${row.name} ${row.resourceType ?? ""} ${row.resourceUri ?? ""}`.toLowerCase();
  if (typeHint === "tiff") {
    return (
      /(?:\.ome\.tiff?|\.tiff?|\.tif)(?:$|\b)/.test(haystack) ||
      /\b(?:ome[-\s]?tiff?|tiff?|tif)\b/.test(haystack)
    );
  }
  if (typeHint === "png") {
    return /(?:\.png)(?:$|\b)/.test(haystack) || /\bpng\b/.test(haystack);
  }
  if (typeHint === "table") {
    return (
      /\btable\b/.test(haystack) ||
      /(?:\.h5|\.hdf5|\.dream3d)(?:$|\b)/.test(haystack) ||
      /\b(?:hdf5|dream3d)\b/.test(haystack)
    );
  }
  if (typeHint === "dataset") {
    return /\bdataset\b/.test(haystack);
  }
  return true;
};

const filterBisqueRowsForPrompt = (
  rows: ToolResourceRow[],
  promptText: string
): ToolResourceRow[] => {
  const typeHint = bisquePromptTypeHint(promptText);
  const hintedRows = rows.filter((row) => bisqueRowMatchesTypeHint(row, typeHint));
  return hintedRows.length > 0 ? hintedRows : rows;
};

const partitionBisqueRowsByUri = (
  rows: ToolResourceRow[]
): { resourceUris: string[]; datasetUris: string[] } => {
  const datasetHints = rows
    .filter((row) => normalizeBisqueServiceKind(row.resourceType) === "dataset")
    .map((row) => String(row.resourceUri ?? row.uri ?? "").trim())
    .filter((uri) => uri.length > 0);
  const uris = rows
    .map((row) => String(row.resourceUri ?? row.uri ?? "").trim())
    .filter((uri) => uri.length > 0);
  return partitionBisqueUris(uris, datasetHints);
};

const hasBisqueSelectionContext = (selectionContext: SelectionContext | null): boolean =>
  Boolean(
    selectionContext &&
      (
        (selectionContext.resource_uris?.length ?? 0) > 0 ||
        (selectionContext.dataset_uris?.length ?? 0) > 0
      )
  );

const isBisqueUploadActionPrompt = (
  promptText: string,
  options?: {
    hasStagedUploads?: boolean;
  }
): boolean => {
  const lowered = String(promptText || "").trim().toLowerCase();
  const hasQueryVerb = /\b(show|see|view|find|search|list|which|what|browse|open|preview)\b/.test(
    lowered
  );
  if (
    hasQueryVerb &&
    /\b(?:recent|latest|most recent|newest)\s+upload\b/.test(lowered)
  ) {
    return false;
  }
  if (hasQueryVerb && /\b(?:my|any|the)\s+uploads?\b/.test(lowered)) {
    return false;
  }
  const hasUploadVerb = /\b(upload|ingest|import)\b/.test(lowered);
  const referencesPayload =
    /\b(this|these|it|them|attached|current|selected)\b/.test(lowered) ||
    Boolean(options?.hasStagedUploads);
  const targetsBisque = /\bbisque\b/.test(lowered) || /\bdataset\b/.test(lowered);
  return hasUploadVerb && (referencesPayload || targetsBisque);
};

const inferBisqueSelectionToolNames = (
  promptText: string,
  options?: {
    hasSelectionContext?: boolean;
    hasStagedUploads?: boolean;
  }
): string[] => {
  const lowered = String(promptText || "").trim().toLowerCase();
  const selected = new Set<string>();
  if (/\bxml\b/.test(lowered)) {
    selected.add("bisque_fetch_xml");
  }
  if (/\b(?:metadata\s+)?tags?\b/.test(lowered)) {
    selected.add("add_tags_to_resource");
  }
  if (
    /\b(?:roi|gobject|annotation|annotations|rectangle|polygon|bounding box|bbox)\b/.test(
      lowered
    )
  ) {
    selected.add("bisque_add_gobjects");
  }
  if (/\bdelete|trash|remove\b/.test(lowered)) {
    selected.add("delete_bisque_resource");
  }
  if (
    /\bdataset\b/.test(lowered) &&
    /\b(download|export|save)\b/.test(lowered)
  ) {
    selected.add("bisque_download_dataset");
  }
  if (
    /\bdataset\b/.test(lowered) &&
    /\b(create|make|build|assemble|call(?:ed)?|named?)\b/.test(lowered)
  ) {
    selected.add("bisque_create_dataset");
  }
  if (
    /\bdataset\b/.test(lowered) &&
    /\b(add|append|put|organize|move|save)\b/.test(lowered)
  ) {
    if (options?.hasStagedUploads || isBisqueUploadActionPrompt(promptText, options)) {
      selected.add("upload_to_bisque");
    } else {
      selected.add("bisque_add_to_dataset");
    }
  }
  if (isBisqueUploadActionPrompt(promptText, options)) {
    selected.add("upload_to_bisque");
  }
  if (/\bdownload\b/.test(lowered)) {
    selected.add("bisque_download_resource");
  }
  const wantsCatalogSearch =
    /\b(do i have|find|search|look for|list|browse|show me|latest|most recent|recent|newest|assets|resources|uploads?)\b/.test(
      lowered
    ) &&
    /\b(png|tiff?|ome[-\s]?tiff?|hdf5|h5|dream3d|image|images|file|files|resource|resources|dataset|datasets|table|tables|upload|uploads)\b/.test(
      lowered
    );
  if (wantsCatalogSearch) {
    selected.add("search_bisque_resources");
  }
  if (
    selected.size === 0 &&
    options?.hasSelectionContext &&
    /\b(show|see|view|preview|open|inspect|metadata|details?|keys?|groups?|datasets?|columns?|headers?|variables?|fields?|schema|structure|layout)\b/.test(
      lowered
    )
  ) {
    selected.add("load_bisque_resource");
  }
  return Array.from(selected);
};

const shouldUseBisqueTargetSelectionContext = (
  promptText: string,
  bisqueUrls: string[],
  options?: {
    hasStagedUploads?: boolean;
  }
): boolean => {
  if (bisqueUrls.length === 0) {
    return false;
  }
  const strippedPrompt = stripBisqueUrls(promptText);
  if (strippedPrompt.length === 0 && !options?.hasStagedUploads) {
    return false;
  }
  const inferredToolNames = inferBisqueSelectionToolNames(promptText, {
    hasSelectionContext: true,
    hasStagedUploads: options?.hasStagedUploads,
  });
  return inferredToolNames.length > 0;
};

const buildBisqueSelectionContext = ({
  source,
  focusedFileIds,
  resourceUris,
  datasetUris,
  artifactHandles,
  originatingUserText,
  suggestedDomain,
  suggestedToolNames,
}: {
  source: SelectionContext["source"];
  focusedFileIds?: string[];
  resourceUris?: string[];
  datasetUris?: string[];
  artifactHandles?: Record<string, string[]>;
  originatingUserText?: string | null;
  suggestedDomain?: SelectionContext["suggested_domain"];
  suggestedToolNames?: string[];
}): SelectionContext => ({
  context_id: makeId(),
  source,
  focused_file_ids: uniqueFileIds(focusedFileIds ?? []),
  resource_uris: Array.from(new Set((resourceUris ?? []).map((value) => String(value || "").trim()).filter(Boolean))),
  dataset_uris: Array.from(new Set((datasetUris ?? []).map((value) => String(value || "").trim()).filter(Boolean))),
  artifact_handles: toArtifactHandleMap(artifactHandles),
  originating_message_id: null,
  originating_user_text: originatingUserText?.trim() || null,
  suggested_domain: suggestedDomain ?? null,
  suggested_tool_names: Array.from(
    new Set((suggestedToolNames ?? []).map((value) => String(value || "").trim()).filter(Boolean))
  ),
});

const deriveBisqueSelectionContextFromToolCards = ({
  toolResultCards,
  source,
  originatingUserText,
  suggestedDomain,
}: {
  toolResultCards: ToolResultCard[];
  source: SelectionContext["source"];
  originatingUserText: string;
  suggestedDomain?: SelectionContext["suggested_domain"];
}): {
  selectionContext: SelectionContext | null;
  resolvedRows: ToolResourceRow[];
  clearsSelection: boolean;
} => {
  const clearsSelection = toolResultCards.some((card) => card.tool === "delete_bisque_resource");
  const resolvedRows = mergeBisqueResourceRows(
    toolResultCards
      .filter((card) => card.tool !== "delete_bisque_resource")
      .flatMap((card) => card.resourceRows)
      .filter((row) => Boolean(row.resourceUri))
  );
  if (resolvedRows.length === 0) {
    return {
      selectionContext: null,
      resolvedRows,
      clearsSelection,
    };
  }
  const partitioned = partitionBisqueRowsByUri(resolvedRows);
  return {
    selectionContext: buildBisqueSelectionContext({
      source,
      resourceUris: partitioned.resourceUris,
      datasetUris: partitioned.datasetUris,
      originatingUserText,
      suggestedDomain,
      suggestedToolNames: [],
    }),
    resolvedRows,
    clearsSelection,
  };
};

const shouldInferBisqueToolsForTurn = (
  promptText: string,
  selectionContext: SelectionContext | null,
  options?: {
    hasStagedUploads?: boolean;
  }
): boolean => {
  const lowered = String(promptText || "").trim().toLowerCase();
  if (!lowered) {
    return false;
  }
  if (/\bbisque\b/.test(lowered) || extractBisqueUrls(promptText).length > 0) {
    return true;
  }
  if (options?.hasStagedUploads && /\b(upload|ingest|import|dataset)\b/.test(lowered)) {
    return true;
  }
  if (!hasBisqueSelectionContext(selectionContext)) {
    return false;
  }
  return /\b(this|that|these|those|selected|current|same|it|them|upload|dataset|download|delete|remove|tag|metadata|annotation|roi|gobject|view|show|open|preview|inspect|search|find|list|latest|most recent)\b/.test(
    lowered
  );
};

const isFreshBisqueDiscoveryPrompt = (promptText: string): boolean => {
  const lowered = String(promptText || "").trim().toLowerCase();
  const typeHint = bisquePromptTypeHint(promptText);
  if (!typeHint) {
    return false;
  }
  if (/\b(this|that|these|those|selected|same)\b/.test(lowered)) {
    return false;
  }
  return /\b(do i have|what about|find|search|look for|latest|most recent)\b/.test(lowered);
};

const inferBisqueReferenceSelection = (
  promptText: string,
  messages: UiMessage[]
): BisqueReferenceSelection | null => {
  const lowered = String(promptText || "").trim().toLowerCase();
  if (isFreshBisqueDiscoveryPrompt(promptText)) {
    return null;
  }
  const desiredCount = parseBisqueSelectionCount(lowered);
  const referencesExistingSelection =
    /\b(these|those|them|correct|selected|this one|that one|this image|that image|this file|that file|this resource|that resource)\b/.test(
      lowered
    ) ||
    /\b(?:this|that)\s+(?:[\w.-]+\s+){0,2}(?:image|file|resource|dataset|result|match)\b/.test(
      lowered
    );
  const explicitSelectionReference =
    referencesExistingSelection ||
    desiredCount !== null ||
    /\b(first|second|third|fourth|fifth|last)\s+(?:one|image|file|resource|result|match)\b/.test(
      lowered
    ) ||
    /\b(latest|most recent)\s+(?:one|image|file|resource|result|match)\b/.test(
      lowered
    );
  const wantsPreview =
    /\b(show|see|view|preview|open)\b/.test(lowered) ||
    /\blooks?\s+like\b/.test(lowered) ||
    /\b(head of|contents of)\b/.test(lowered) ||
    /\b(keys?|groups?|datasets?|columns?|headers?|variables?|fields?|schema|structure|layout)\b/.test(
      lowered
    );
  const wantsSelectionAction =
    /\b(add to chat|use in chat|bring (?:it|them) into chat|chat context|work with it here|download)\b/.test(
      lowered
    );
  const wantsResourceMutation =
    /\b(delete|trash|remove|tag|tags|metadata tag|annotation|annotations|roi|gobject|rectangle|polygon|bounding box|bbox)\b/.test(
      lowered
    );
  const wantsDataset =
    /\bdataset\b/.test(lowered) &&
    /\b(make|create|build|assemble|call(?:ed)?|named?)\b/.test(lowered);
  if (!wantsPreview && !wantsDataset && !wantsSelectionAction && !wantsResourceMutation) {
    return null;
  }
  if (
    (wantsPreview || wantsSelectionAction || wantsResourceMutation) &&
    !explicitSelectionReference &&
    !wantsDataset
  ) {
    return null;
  }

  const assistantMessages = [...messages].reverse().filter((message) => message.role === "assistant");
  let fallbackRows: ToolResourceRow[] = [];
  for (const message of assistantMessages) {
    const resolvedRows = extractResolvedBisqueRowsFromMessage(message);
    const candidateResolvedRows = filterBisqueRowsForPrompt(resolvedRows, promptText);
    if (referencesExistingSelection && candidateResolvedRows.length > 0) {
      fallbackRows = candidateResolvedRows;
      break;
    }
    const searchRows = extractSearchResourceRowsFromMessage(message);
    const candidateSearchRows = filterBisqueRowsForPrompt(searchRows, promptText);
    if (candidateSearchRows.length > 0) {
      fallbackRows = candidateSearchRows;
      break;
    }
    if (fallbackRows.length === 0) {
      fallbackRows = candidateResolvedRows.length > 0 ? candidateResolvedRows : candidateSearchRows;
    }
  }

  if (fallbackRows.length === 0) {
    return null;
  }

  const defaultCount =
    referencesExistingSelection || wantsDataset
      ? fallbackRows.length
      : 1;
  const selectedRows = fallbackRows.slice(0, desiredCount && desiredCount > 0 ? desiredCount : defaultCount);

  if (selectedRows.length === 0) {
    return null;
  }

  return {
    sourceRows: fallbackRows,
    selectedRows,
    intent: wantsPreview ? "preview" : "selection",
  };
};

const thinkingBarTextForRunEvents = (
  runEvents: RunEvent[],
  isStreaming: boolean
): string | null => {
  if (!isStreaming) {
    return null;
  }
  for (let index = runEvents.length - 1; index >= 0; index -= 1) {
    const event = runEvents[index];
    const payload = toRecord(event.payload);
    if (!payload) {
      continue;
    }
    const eventType = String(event.event_type || "").trim().toLowerCase();
    const phase = String(payload.phase || "").trim().toLowerCase();
    const status = String(payload.status || "").trim().toLowerCase();
    if (eventType === "memory.retrieved" || phase === "memory") {
      return getPhaseThinkingText("memory") ?? DEFAULT_THINKING_TEXT;
    }
    if (eventType === "knowledge.retrieved" || phase === "knowledge") {
      return getPhaseThinkingText("knowledge") ?? DEFAULT_THINKING_TEXT;
    }
    if (eventType === "learning.promoted" || eventType === "learning.skipped" || phase === "learning") {
      return getPhaseThinkingText("learning") ?? DEFAULT_THINKING_TEXT;
    }
    if (
      eventType === "tool_event" ||
      eventType === "pro_mode.tool_requested" ||
      eventType === "pro_mode.tool_completed"
    ) {
      const toolStatus =
        status ||
        (eventType === "pro_mode.tool_requested"
          ? "started"
          : eventType === "pro_mode.tool_completed"
            ? "completed"
            : "");
      return getToolStatusThinkingText(toolStatus) ?? DEFAULT_THINKING_TEXT;
    }
    if (
      eventType !== "graph_event" &&
      eventType !== "pro_mode.phase_started" &&
      eventType !== "pro_mode.phase_completed" &&
      eventType !== "pro_mode.convergence_updated" &&
      eventType !== "pro_mode.verifier_result"
    ) {
      continue;
    }
    const phaseThinkingText = getPhaseThinkingText(phase);
    if (phaseThinkingText) {
      return phaseThinkingText;
    }
  }
  return isStreaming ? DEFAULT_THINKING_TEXT : null;
};

const summaryModeLabelForMessage = (message: UiMessage): string | null => {
  const metadata = toRecord(message.responseMetadata);
  const debug = metadata ? toRecord(metadata.debug) : null;
  const proMode = metadata ? toRecord(metadata.pro_mode) : null;
  const path = String(debug?.path || "").trim().toLowerCase();
  return path === "pro_mode" || Boolean(proMode) ? "Pro Mode" : null;
};

const proModeDevConversationForMessage = (
  message: UiMessage
): Record<string, unknown> | null => {
  if (!import.meta.env.DEV) {
    return null;
  }
  const metadata = toRecord(message.responseMetadata);
  if (!metadata) {
    return null;
  }
  const debug = toRecord(metadata.debug);
  if (String(debug?.path || "").trim().toLowerCase() !== "pro_mode") {
    return null;
  }
  const proMode = toRecord(metadata.pro_mode);
  const conversation = toRecord(proMode?.dev_conversation);
  return conversation;
};

const formatReasoningDuration = (seconds: number | null | undefined): string | null => {
  const value = Number(seconds ?? 0);
  if (!Number.isFinite(value) || value <= 0) {
    return null;
  }
  if (value < 10) {
    return `${value.toFixed(1)}s`;
  }
  if (value < 60) {
    return `${value.toFixed(0)}s`;
  }
  if (value < 3600) {
    return `${(value / 60).toFixed(1)}m`;
  }
  return `${(value / 3600).toFixed(1)}h`;
};

const toolCardImagesFromBisqueResourceRows = (
  rows: ToolResourceRow[],
  limit: number = 6
): ToolCardImage[] => {
  const images: ToolCardImage[] = [];
  const seen = new Set<string>();
  rows.forEach((row) => {
    if (images.length >= limit) {
      return;
    }
    const previewUrl = buildBisqueThumbnailUrl(row.imageServiceUrl);
    if (!previewUrl) {
      return;
    }
    const key =
      row.resourceUri?.toLowerCase() ||
      row.clientViewUrl?.toLowerCase() ||
      row.uri?.toLowerCase() ||
      previewUrl.toLowerCase();
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    images.push({
      path: `${key}#bisque-preview`,
      url: previewUrl,
      downloadUrl: row.imageServiceUrl || row.clientViewUrl || row.resourceUri || undefined,
      title: row.name || "BisQue preview",
      sourceName: row.name || row.resourceUri || row.clientViewUrl || "bisque-resource",
      previewable: true,
    });
  });
  return images;
};

const toolCardImagesFromUploadedMatches = (
  rawFiles: string[],
  uploadedPreviewLookup: Map<string, UploadedFileRecord[]>,
  buildUploadPreviewUrl: (fileId: string) => string,
  limit: number = 6
): ToolCardImage[] => {
  const images: ToolCardImage[] = [];
  const seen = new Set<string>();
  rawFiles.forEach((rawFile) => {
    if (images.length >= limit) {
      return;
    }
    const matchedUpload = resolveUploadedArtifactPreview(rawFile, uploadedPreviewLookup);
    if (!matchedUpload) {
      return;
    }
    const key = matchedUpload.file_id.toLowerCase();
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    images.push({
      path: `uploaded:${matchedUpload.file_id}`,
      url: buildUploadPreviewUrl(matchedUpload.file_id),
      downloadUrl:
        matchedUpload.client_view_url ||
        matchedUpload.canonical_resource_uri ||
        undefined,
      title: matchedUpload.original_name,
      sourceName: rawFile || matchedUpload.original_name,
      previewable: true,
      linkedFileId: matchedUpload.file_id,
    });
  });
  return images;
};

const uploadedPreviewArtifactFromPath = (
  rawFile: string,
  uploadedPreviewLookup: Map<string, UploadedFileRecord[]>,
  buildUploadPreviewUrl: (fileId: string) => string
): RunImageArtifact | undefined => {
  const matchedUpload = resolveUploadedArtifactPreview(rawFile, uploadedPreviewLookup);
  if (!matchedUpload) {
    return undefined;
  }
  return {
    path: `uploaded:${matchedUpload.file_id}`,
    url: buildUploadPreviewUrl(matchedUpload.file_id),
    downloadUrl:
      matchedUpload.client_view_url ||
      matchedUpload.canonical_resource_uri ||
      undefined,
    title: matchedUpload.original_name,
    sourceName: rawFile || matchedUpload.original_name,
    previewable: true,
    linkedFileId: matchedUpload.file_id,
  } satisfies RunImageArtifact;
};

const pluralizeCount = (count: number, singular: string, plural?: string): string =>
  `${count} ${count === 1 ? singular : plural ?? `${singular}s`}`;

const normalizeBisqueServiceKind = (
  value: string | null | undefined
): "image" | "table" | "dataset" | "resource" => {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (
    normalized === "image" ||
    normalized === "image_service" ||
    normalized === "file"
  ) {
    return "image";
  }
  if (normalized === "table") {
    return "table";
  }
  if (normalized === "dataset") {
    return "dataset";
  }
  return "resource";
};

const parseYoloClassCounts = (value: unknown): YoloFigureClassCount[] => {
  if (!value || typeof value !== "object") {
    return [];
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => toRecord(item))
      .filter((item): item is Record<string, unknown> => item !== null)
      .map((item) => ({
        name: String(item.class_name ?? item.name ?? "class").trim() || "class",
        count: Math.max(0, Math.round(toNumber(item.count) ?? 0)),
      }))
      .filter((item) => item.count > 0)
      .slice(0, 8);
  }
  return Object.entries(value)
    .map(([name, count]) => ({
      name: String(name).trim() || "class",
      count: Math.max(0, Math.round(toNumber(count) ?? 0)),
    }))
    .filter((item) => item.count > 0)
    .slice(0, 8);
};

const resolveArtifactForLookup = (
  value: string,
  artifactBySource: Map<string, RunImageArtifact[]>
): RunImageArtifact | undefined => {
  for (const key of artifactLookupKeys(value)) {
    const matches = artifactBySource.get(key);
    if (matches && matches.length > 0) {
      return matches[0];
    }
  }
  return undefined;
};

const isMatplotlibAnnotatedArtifact = (artifact: RunImageArtifact): boolean =>
  /matplotlib_annotated/i.test(`${artifact.path} ${artifact.sourceName}`);

const yoloAnnotatedArtifactOrder = (artifact: RunImageArtifact): [number, number, string] => {
  const normalized = `${artifact.path} ${artifact.sourceName}`.toLowerCase();
  const indexedMatch = normalized.match(/(?:^|__)(\d{3,4})-/);
  if (indexedMatch) {
    return [0, Number(indexedMatch[1]), normalized];
  }
  return [1, Number.MAX_SAFE_INTEGER, normalized];
};

const prairieAnalysisClassCounts = (
  analysis: PrairieImageAnalysis | undefined
): YoloFigureClassCount[] => {
  if (!analysis) {
    return [];
  }
  return [
    {
      name: "prairie_dog",
      count: Math.max(0, Math.round(Number(analysis.prairieDogCount ?? 0))),
    },
    {
      name: "burrow",
      count: Math.max(0, Math.round(Number(analysis.burrowCount ?? 0))),
    },
  ].filter((item) => item.count > 0);
};

const buildYoloFigureCards = (
  records: unknown[],
  artifactBySource: Map<string, RunImageArtifact[]>,
  uploadedPreviewLookup: Map<string, UploadedFileRecord[]>,
  buildUploadPreviewUrl: (fileId: string) => string
): YoloFigureCard[] => {
  return records
    .map((item, index): YoloFigureCard | null => {
      const row = toRecord(item);
      if (!row) {
        return null;
      }
      const sourcePath = String(row.source_path ?? row.path ?? "").trim();
      const sourceName = String(row.source_name ?? "").trim();
      const previewPath = String(row.preview_path ?? "").trim();
      const previewName = String(row.preview_name ?? "").trim();
      const rawSourcePath = String(row.raw_source_path ?? sourcePath ?? "").trim();
      const rawSourceName = String(row.raw_source_name ?? sourceName ?? "").trim();
      const previewKind = String(row.preview_kind ?? "").trim();
      const allowsOriginalDisplayFallback =
        previewKind === "original_fallback" ||
        (previewPath.length > 0 && previewPath === rawSourcePath);
      const previewArtifact =
        resolveArtifactForLookup(previewPath, artifactBySource) ??
        resolveArtifactForLookup(previewName, artifactBySource) ??
        (allowsOriginalDisplayFallback
          ? resolveArtifactForLookup(sourcePath, artifactBySource) ??
            resolveArtifactForLookup(sourceName, artifactBySource) ??
            uploadedPreviewArtifactFromPath(
              previewPath || sourcePath,
              uploadedPreviewLookup,
              buildUploadPreviewUrl
            )
          : undefined);
      const rawArtifact =
        resolveArtifactForLookup(rawSourcePath, artifactBySource) ??
        resolveArtifactForLookup(sourcePath, artifactBySource) ??
        resolveArtifactForLookup(rawSourceName, artifactBySource) ??
        resolveArtifactForLookup(sourceName, artifactBySource) ??
        uploadedPreviewArtifactFromPath(
          rawSourcePath || sourcePath,
          uploadedPreviewLookup,
          buildUploadPreviewUrl
        );
      const displayedArtifact = previewArtifact ?? (allowsOriginalDisplayFallback ? rawArtifact : undefined);
      if (!displayedArtifact) {
        return null;
      }
      const classCounts = parseYoloClassCounts(row.class_counts);
      const boxCount = toNumber(row.box_count);
      const imageWidth = toNumber(row.image_width);
      const imageHeight = toNumber(row.image_height);
      const title =
        toDisplayFileLabel(
          rawSourceName ||
            sourceName ||
            previewName ||
            displayedArtifact.sourceName ||
            artifactTitleFromPath(previewPath || sourcePath || displayedArtifact.path)
        ) ||
        rawSourceName ||
        sourceName ||
        previewName ||
        artifactTitleFromPath(previewPath || sourcePath || displayedArtifact.path) ||
        `Detection ${index + 1}`;
      const subtitle = [
        previewKind === "original_fallback" ? "Original image" : null,
        imageWidth !== null && imageHeight !== null
          ? `${Math.round(imageWidth)} × ${Math.round(imageHeight)} px`
          : null,
      ]
        .filter((value): value is string => value !== null)
        .join(" · ");
      const figure: YoloFigureCard = {
        key: `${sourcePath || previewPath || displayedArtifact.path}-${index}`,
        title,
        subtitle,
        previewUrl: displayedArtifact.url,
        downloadUrl: displayedArtifact.downloadUrl ?? displayedArtifact.url,
        originalUrl: rawArtifact?.downloadUrl ?? rawArtifact?.url,
        previewKind: previewKind || undefined,
        sourceName: sourceName || undefined,
        rawSourceName: rawSourceName || undefined,
        sourcePath: sourcePath || undefined,
        rawSourcePath: rawSourcePath || undefined,
        imageWidth,
        imageHeight,
        boxCount,
        classCounts,
        previewable: displayedArtifact.previewable,
      };
      return figure;
    })
    .filter((item): item is YoloFigureCard => item !== null);
};

const buildYoloFigureCardsFromAnnotatedArtifacts = (
  annotatedArtifacts: RunImageArtifact[],
  recordHints: unknown[],
  predictionImageRawPaths: string[],
  prairieImageAnalyses: PrairieImageAnalysis[],
  artifactBySource: Map<string, RunImageArtifact[]>,
  uploadedPreviewLookup: Map<string, UploadedFileRecord[]>,
  buildUploadPreviewUrl: (fileId: string) => string
): YoloFigureCard[] => {
  const sortedArtifacts = [...annotatedArtifacts].sort((left, right) => {
    const leftKey = yoloAnnotatedArtifactOrder(left);
    const rightKey = yoloAnnotatedArtifactOrder(right);
    return (
      leftKey[0] - rightKey[0] ||
      leftKey[1] - rightKey[1] ||
      leftKey[2].localeCompare(rightKey[2])
    );
  });

  return sortedArtifacts
    .map((artifact, figureIndex): YoloFigureCard | null => {
      const recordHint = toRecord(recordHints[figureIndex]);
      const prairieAnalysis = prairieImageAnalyses[figureIndex];
      const rawSourcePath = String(
        recordHint?.raw_source_path ??
          predictionImageRawPaths[figureIndex] ??
          prairieAnalysis?.rawFile ??
          ""
      ).trim();
      const rawSourceName = String(
        recordHint?.raw_source_name ?? extractFilename(rawSourcePath)
      ).trim();
      const rawArtifact =
        resolveArtifactForLookup(rawSourcePath, artifactBySource) ??
        uploadedPreviewArtifactFromPath(
          rawSourcePath,
          uploadedPreviewLookup,
          buildUploadPreviewUrl
        );
      const recordClassCounts = parseYoloClassCounts(recordHint?.class_counts);
      const classCounts =
        recordClassCounts.length > 0
          ? recordClassCounts
          : prairieAnalysisClassCounts(prairieAnalysis);
      const boxCount = toNumber(recordHint?.box_count) ?? prairieAnalysis?.boxCount ?? null;
      const imageWidth = toNumber(recordHint?.image_width);
      const imageHeight = toNumber(recordHint?.image_height);
      const title =
        toDisplayFileLabel(rawSourceName || artifact.sourceName) ||
        rawSourceName ||
        artifact.sourceName ||
        artifactTitleFromPath(artifact.path) ||
        `Detection ${figureIndex + 1}`;
      const subtitle =
        imageWidth !== null && imageHeight !== null
          ? `${Math.round(imageWidth)} × ${Math.round(imageHeight)} px`
          : undefined;

      return {
        key: `${artifact.path}-${figureIndex}`,
        title,
        subtitle,
        previewUrl: artifact.url,
        downloadUrl: artifact.downloadUrl ?? artifact.url,
        originalUrl: rawArtifact?.downloadUrl ?? rawArtifact?.url,
        previewKind: "matplotlib_annotated",
        sourceName: artifact.sourceName || undefined,
        rawSourceName: rawSourceName || undefined,
        sourcePath: artifact.sourcePath || artifact.path,
        rawSourcePath: rawSourcePath || undefined,
        imageWidth,
        imageHeight,
        boxCount,
        classCounts,
        previewable: artifact.previewable,
      } satisfies YoloFigureCard;
    })
    .filter((item): item is YoloFigureCard => item !== null);
};

function PanelLoadingState({
  title = "Loading panel...",
  subtitle = "Preparing this workspace only when you open it keeps the chat shell lighter.",
}: {
  title?: string;
  subtitle?: string;
}) {
  return (
    <div className="hero-state">
      <h2 className="hero-title">{title}</h2>
      <p className="hero-subtitle">{subtitle}</p>
    </div>
  );
}

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

type SidebarAccountSettingsButtonProps = {
  authUser: string | null;
  authMode: AuthMode | null;
  authIsAdmin: boolean;
  themePreference: ThemePreference;
  onThemePreferenceChange: (value: ThemePreference) => void;
  onOpenSettings: () => void;
  onLogout: () => Promise<void>;
};

function SidebarAccountSettingsButton({
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

export function App() {
  const apiBaseUrl = DEFAULT_API_BASE_URL;
  const [themePreference, setThemePreference] = useLocalStorageState<ThemePreference>(
    "bisque.frontend.themePreference",
    "system"
  );
  const [resolvedTheme, setResolvedTheme] = useState<"light" | "dark">("light");
  const [bisqueNavLinks, setBisqueNavLinks] = useState<BisqueNavLinks | null>(() => {
    const fallbackRoot = inferBisqueRootFromUrl(DEFAULT_BISQUE_BROWSER_URL);
    return fallbackRoot ? buildBisqueNavLinks(fallbackRoot) : null;
  });
  const [bisqueResourceCountsState, setBisqueResourceCountsState] =
    useState<BisqueResourceCountsState>({ requestKey: "", counts: null });
  const [authStatus, setAuthStatus] = useState<AuthStatus>("checking");
  const [authUser, setAuthUser] = useState<string | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode | null>(null);
  const [authProvider, setAuthProvider] = useState<AuthProvider>("local");
  const [authIsAdmin, setAuthIsAdmin] = useState(false);
  const [bisqueCredentialsLinked, setBisqueCredentialsLinked] = useState(false);
  const [authError, setAuthError] = useState<string | null>(() => readAuthErrorFromLocation());
  const [authNotice, setAuthNotice] = useState<string | null>(null);
  const [authSubmitting, setAuthSubmitting] = useState(false);
  const [authGuestEnabled, setAuthGuestEnabled] = useState(true);
  const hostedAuthRedirectAttemptedRef = useRef(false);
  const sessionRevalidatedAtRef = useRef(0);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [blankChatTokenUsage, setBlankChatTokenUsage] =
    useState<TokenUsageResponse | null>(null);
  const [blankChatUsageLoading, setBlankChatUsageLoading] = useState(false);
  const [blankChatUsageError, setBlankChatUsageError] = useState<string | null>(null);
  const blankChatUsageRequestRef = useRef({
    key: "",
    inFlight: false,
    loaded: false,
    failed: false,
  });
  const isPhoneView = useBreakpoint(641);

  const [conversations, setConversations] = useState<ConversationState[]>([]);
  const [conversationListOffset, setConversationListOffset] = useState(0);
  const [conversationListHasMore, setConversationListHasMore] = useState(false);
  const [conversationListLoadingMore, setConversationListLoadingMore] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [conversationsHydrated, setConversationsHydrated] = useState(false);
  const [activePanel, setActivePanel] = useState<ActivePanel>("chat");
  const [viewerOpen, setViewerOpen] = useState(false);
  const [conversationDeletingById, setConversationDeletingById] = useState<
    Record<string, boolean>
  >({});
  const [pendingConversationDelete, setPendingConversationDelete] =
    useState<PendingConversationDelete | null>(null);
  const [conversationRenamingById, setConversationRenamingById] = useState<
    Record<string, boolean>
  >({});
  const [pendingConversationRename, setPendingConversationRename] =
    useState<PendingConversationRename | null>(null);
  const [resourceViewerContext, setResourceViewerContext] = useState<ResourceViewerContext | null>(
    null
  );
  const [resources, setResources] = useState<ResourceRecord[]>([]);
  const [resourceCollections, setResourceCollections] = useState<ResourceCollectionRecord[]>([]);
  const [activeResourceCollectionId, setActiveResourceCollectionId] = useState<string | null>(null);
  const [activeResourceCollectionSnapshot, setActiveResourceCollectionSnapshot] =
    useState<ResourceCollectionRecord | null>(null);
  const [resourceTotalCount, setResourceTotalCount] = useState(0);
  const [resourcesLoading, setResourcesLoading] = useState(false);
  const [resourcesLoadingMore, setResourcesLoadingMore] = useState(false);
  const [resourceCollectionsLoading, setResourceCollectionsLoading] = useState(false);
  const [resourcesError, setResourcesError] = useState<string | null>(null);
  const [resourcesUploading, setResourcesUploading] = useState(false);
  const [resourceUploadProgress, setResourceUploadProgress] = useState<ResourceUploadProgress[]>(
    () => readResourceUploadProgressFromStorage()
  );
  const resourceUploadProgressBatcherRef =
    useRef<ReturnType<typeof createResourceUploadProgressFrameBatcher> | null>(null);
  const inFlightUploadCount = useMemo(
    () => summarizeResourceUploadProgress(resourceUploadProgress).inFlight,
    [resourceUploadProgress]
  );
  if (resourceUploadProgressBatcherRef.current === null) {
    resourceUploadProgressBatcherRef.current = createResourceUploadProgressFrameBatcher({
      onFlush: (events) => {
        setResourceUploadProgress((current) => {
          const next = events.reduce(
            (items, event) => mergeResourceUploadProgress(items, event),
            current
          );
          writeResourceUploadProgressToStorage(next);
          const { inFlight, completed, failed } = summarizeResourceUploadProgress(next);
          const previousInFlight = resourceUploadInFlightRef.current;
          resourceUploadInFlightRef.current = inFlight;
          // Drain edge: a batch just settled (in-flight went from >0 to 0).
          if (previousInFlight > 0 && inFlight === 0 && completed + failed > 0) {
            if (failed === 0) {
              showSuccessToast(
                `${completed} upload${completed === 1 ? "" : "s"} finished`
              );
            } else if (completed === 0) {
              showErrorToast(`${failed} upload${failed === 1 ? "" : "s"} failed`);
            } else {
              showErrorToast(`${completed} finished, ${failed} failed`);
            }
          }
          return next;
        });
      },
    });
  }
  const resourceUploadInFlightRef = useRef(0);
  const pausedResourceUploadSessionIdsRef = useRef<Set<string>>(new Set());
  const resourceListKeyRef = useRef("");
  const [mobileConversationQuery, setMobileConversationQuery] = useState("");
  const [resourceQuery, setResourceQuery] = useState("");
  const [debouncedResourceQuery, setDebouncedResourceQuery] = useState("");
  const [composerResourceQuery, setComposerResourceQuery] = useState("");
  const [composerResources, setComposerResources] = useState<ResourceRecord[]>([]);
  const [composerResourcesLoading, setComposerResourcesLoading] = useState(false);
  const [composerResourcesError, setComposerResourcesError] = useState<string | null>(null);
  const [composerResourcePickerOpen, setComposerResourcePickerOpen] = useState(false);
  const [activeComposerResourceId, setActiveComposerResourceId] = useState<string | null>(null);
  const [composerResourcePickerSelection, setComposerResourcePickerSelection] = useState<
    Record<string, ResourceRecord>
  >({});
  const [resourceKindFilter, setResourceKindFilter] = useState<ResourceKindFilter>("all");
  const [resourceSourceFilter, setResourceSourceFilter] =
    useState<ResourceSourceFilter>("all");
  const [resourceSharingFilter, setResourceSharingFilter] =
    useState<ResourceSharingFilter>("all");
  const [resourceStatusFilter, setResourceStatusFilter] =
    useState<ResourceStatusFilter>("active");
  const [resourceTagFilter, setResourceTagFilter] = useState("");
  const [resourceRefreshToken, setResourceRefreshToken] = useState(0);
  const [resourceCollectionRefreshToken, setResourceCollectionRefreshToken] = useState(0);
  const [resourceDeletingById, setResourceDeletingById] = useState<Record<string, boolean>>({});
  const [resourceRestoringById, setResourceRestoringById] = useState<Record<string, boolean>>({});
  const [resourceCollectionRestoringById, setResourceCollectionRestoringById] = useState<
    Record<string, boolean>
  >({});
  const [pendingResourceDelete, setPendingResourceDelete] = useState<ResourceRecord | null>(null);
  const [pendingBulkResourceDelete, setPendingBulkResourceDelete] = useState<ResourceRecord[]>([]);
  const [adminOverview, setAdminOverview] = useState<AdminOverviewResponse | null>(null);
  const [adminOrganizations, setAdminOrganizations] = useState<AdminOrganization[]>([]);
  const [adminUsers, setAdminUsers] = useState<AdminUserSummary[]>([]);
  const [adminRuns, setAdminRuns] = useState<AdminRunRecord[]>([]);
  const [adminIssues, setAdminIssues] = useState<AdminIssueRecord[]>([]);
  const [adminLoadingOverview, setAdminLoadingOverview] = useState(false);
  const [adminLoadingOrganizations, setAdminLoadingOrganizations] = useState(false);
  const [adminLoadingUsers, setAdminLoadingUsers] = useState(false);
  const [adminLoadingRuns, setAdminLoadingRuns] = useState(false);
  const [adminLoadingIssues, setAdminLoadingIssues] = useState(false);
  const [adminError, setAdminError] = useState<string | null>(null);
  const [adminRunStatusFilter, setAdminRunStatusFilter] = useState("running");
  const [adminRunQuery, setAdminRunQuery] = useState("");
  const [adminUserQuery, setAdminUserQuery] = useState("");
  const [adminRefreshToken, setAdminRefreshToken] = useState(0);
  const [activeAdminRunEventRunId, setActiveAdminRunEventRunId] = useState<string | null>(null);
  const [adminRunEventsById, setAdminRunEventsById] = useState<Record<string, RunEvent[]>>({});
  const [adminRunEventsLoadingById, setAdminRunEventsLoadingById] = useState<
    Record<string, boolean>
  >({});

  const [adminRunCancellingById, setAdminRunCancellingById] = useState<Record<string, boolean>>(
    {}
  );
  const [adminRunRequeueingById, setAdminRunRequeueingById] = useState<Record<string, boolean>>(
    {}
  );
  const [adminUserDeletingById, setAdminUserDeletingById] = useState<Record<string, boolean>>({});
  const [adminUserUpdatingById, setAdminUserUpdatingById] = useState<Record<string, boolean>>({});
  const [adminDeletingConversationKey, setAdminDeletingConversationKey] = useState<string | null>(
    null
  );
  const [uiErrorBanner, setUiErrorBanner] = useState<string | null>(null);
  const [pendingReusePrompt, setPendingReusePrompt] = useState<PendingReusePrompt | null>(null);
  const [composerDraftsByConversationId, setComposerDraftsByConversationId] = useState<
    Record<string, string>
  >(() => readComposerDraftsFromStorage());

  const sidebarInsetRef = useRef<HTMLElement>(null);
  const [sidebarInsetElement, setSidebarInsetElement] = useState<HTMLElement | null>(null);
  const setSidebarInsetNode = useCallback((node: HTMLElement | null): void => {
    sidebarInsetRef.current = node;
    setSidebarInsetElement(node);
  }, []);
  const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const activeChatScrollElementRef = useRef<HTMLElement | null>(null);
  const conversationScrollMemoryRef = useRef<Record<string, ConversationScrollMemory>>({});
  const conversationScrollWriteBlockRef = useRef<string | null>(null);
  const persistedConversationHashesRef = useRef<Record<string, string>>({});
  const optimisticConversationIdsRef = useRef<Set<string>>(new Set());
  const hydratingConversationIdsRef = useRef<Set<string>>(new Set());
  const activeChatAbortControllersRef = useRef<Map<string, AbortController>>(new Map());
  const runStreamRecoveryControllersRef = useRef<Map<string, AbortController>>(new Map());
  const stopRequestedConversationIdsRef = useRef<Set<string>>(new Set());
  const copyFeedbackTimeoutRef = useRef<number | null>(null);
  const reuseDecisionResolverRef = useRef<((decision: ReuseDecision) => void) | null>(null);
  const runArtifactHydrationsRef = useRef<Set<string>>(new Set());
  const [activeSlashWorkflowId, setActiveSlashWorkflowId] = useState<ComposerWorkflowId | null>(
    null
  );
  const [composerWorkflows, setComposerWorkflows] =
    useState<ComposerWorkflowsModule | null>(null);
  const [dismissedSlashPrompt, setDismissedSlashPrompt] = useState<string | null>(null);
  const [chatScrollRequestKey, setChatScrollRequestKey] = useState(0);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  const apiClient = useMemo(
    () => new ApiClient({ baseUrl: apiBaseUrl, apiKey: DEFAULT_API_KEY }),
    [apiBaseUrl]
  );
  useEffect(() => {
    let cancelled = false;
    void hydrateResourceUploadProgressFromQueueStore(resourceUploadQueueStore, {
      loadUploadSession: (sessionId) => apiClient.getUploadSessionStatus(sessionId),
    })
      .then((hydratedProgress) => {
        if (cancelled || hydratedProgress.length === 0) {
          return;
        }
        setResourceUploadProgress((current) => {
          const hydratedIds = new Set(hydratedProgress.map((item) => item.id));
          const next = [
            ...hydratedProgress,
            ...current.filter((item) => !hydratedIds.has(item.id)),
          ].slice(0, 12);
          writeResourceUploadProgressToStorage(next);
          return next;
        });
      })
      .catch(() => {
        // Queue hydration is best effort; server upload sessions stay authoritative.
      });
    return () => {
      cancelled = true;
    };
  }, [apiClient]);
  useEffect(
    () => () => {
      resourceUploadProgressBatcherRef.current?.clear();
    },
    []
  );
  const activeResourceCollection = useMemo(() => {
    if (!activeResourceCollectionId) {
      return null;
    }
    return (
      resourceCollections.find(
        (collection) => collection.collection_id === activeResourceCollectionId
      ) ?? activeResourceCollectionSnapshot
    );
  }, [activeResourceCollectionId, activeResourceCollectionSnapshot, resourceCollections]);
  const bisqueResourceCountsRequestKey = useMemo(() => {
    if (
      authStatus !== "authenticated" ||
      (authMode !== "bisque" && authMode !== "workos") ||
      !bisqueCredentialsLinked ||
      !bisqueNavLinks
    ) {
      return "";
    }
    return [
      authStatus,
      authMode,
      bisqueCredentialsLinked ? "linked" : "unlinked",
      bisqueNavLinks.home,
      bisqueNavLinks.images,
      bisqueNavLinks.datasets,
      bisqueNavLinks.tables,
    ].join("\u0000");
  }, [authMode, authStatus, bisqueCredentialsLinked, bisqueNavLinks]);
  const bisqueResourceCounts =
    bisqueResourceCountsState.requestKey === bisqueResourceCountsRequestKey
      ? bisqueResourceCountsState.counts
      : null;

  useEffect(() => {
    if (!bisqueResourceCountsRequestKey) {
      return;
    }
    if (bisqueResourceCountsState.requestKey === bisqueResourceCountsRequestKey) {
      return;
    }

    let isCancelled = false;

    void Promise.all([
      apiClient.searchBisqueResources({
        resourceType: "image",
        scope: "owner",
        limit: 1,
        countAll: true,
      }),
      apiClient.searchBisqueResources({
        resourceType: "dataset",
        scope: "owner",
        limit: 1,
        countAll: true,
      }),
      apiClient.searchBisqueResources({
        resourceType: "table",
        scope: "owner",
        limit: 1,
        countAll: true,
      }),
    ])
      .then(([imageSearch, datasetSearch, tableSearch]) => {
        if (!isCancelled) {
          setBisqueResourceCountsState({
            requestKey: bisqueResourceCountsRequestKey,
            counts: {
              image: Math.max(0, Math.floor(Number(imageSearch.count) || 0)),
              dataset: Math.max(0, Math.floor(Number(datasetSearch.count) || 0)),
              table: Math.max(0, Math.floor(Number(tableSearch.count) || 0)),
            },
          });
        }
      })
      .catch(() => {
        if (!isCancelled) {
          setBisqueResourceCountsState({
            requestKey: bisqueResourceCountsRequestKey,
            counts: null,
          });
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [apiClient, bisqueResourceCountsRequestKey, bisqueResourceCountsState.requestKey]);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      try {
        window.localStorage.setItem(
          COMPOSER_DRAFTS_STORAGE_KEY,
          JSON.stringify(composerDraftsByConversationId)
        );
      } catch {
        // Ignore local storage write failures for unsent drafts.
      }
    }, 250);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [composerDraftsByConversationId]);

  const hashHydratedConversations = useCallback(
    (items: ConversationState[]): Record<string, string> =>
      Object.fromEntries(
        items
          .filter((conversation) => conversation.hydrated)
          .map((conversation) => {
            const record = conversationToRecord(conversation);
            return [conversation.id, JSON.stringify(record)];
          })
      ),
    []
  );

  const ensureConversationHydrated = useCallback(
    async (conversationId: string): Promise<void> => {
      const normalizedConversationId = String(conversationId || "").trim();
      if (!normalizedConversationId) {
        return;
      }
      const currentConversation = conversations.find(
        (conversation) => conversation.id === normalizedConversationId
      );
      if (!currentConversation || currentConversation.hydrated) {
        return;
      }
      if (hydratingConversationIdsRef.current.has(normalizedConversationId)) {
        return;
      }
      hydratingConversationIdsRef.current.add(normalizedConversationId);
      try {
        const record = await apiClient.getConversation(normalizedConversationId);
        const hydratedConversation = conversationFromRecord(record);
        setConversations((previous) =>
          mergeConversationPage(previous, [hydratedConversation])
        );
      } catch (error) {
        setUiErrorBanner(`Failed to load chat: ${normalizeApiError(error)}`);
      } finally {
        hydratingConversationIdsRef.current.delete(normalizedConversationId);
      }
    },
    [apiClient, conversations]
  );

  const loadMoreConversations = useCallback(async (): Promise<void> => {
    if (conversationListLoadingMore || !conversationListHasMore || authStatus !== "authenticated") {
      return;
    }
    setConversationListLoadingMore(true);
    try {
      const payload = await listSessionConversations(apiClient, {
        limit: CONVERSATION_PAGE_SIZE,
        offset: conversationListOffset,
      });
      const nextConversations = payload.conversations.map(conversationFromRecord);
      setConversations((previous) => mergeConversationPage(previous, nextConversations));
      setConversationListOffset(payload.offset + payload.count);
      setConversationListHasMore(payload.has_more);
    } catch (error) {
      setUiErrorBanner(`Failed to load more chats: ${normalizeApiError(error)}`);
    } finally {
      setConversationListLoadingMore(false);
    }
  }, [
    apiClient,
    authStatus,
    conversationListHasMore,
    conversationListLoadingMore,
    conversationListOffset,
  ]);

  useEffect(() => {
    return () => {
      if (reuseDecisionResolverRef.current) {
        reuseDecisionResolverRef.current("rerun");
        reuseDecisionResolverRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const activeChatAbortControllers = activeChatAbortControllersRef.current;
    const runStreamRecoveryControllers = runStreamRecoveryControllersRef.current;
    const stopRequestedConversationIds = stopRequestedConversationIdsRef.current;
    return () => {
      activeChatAbortControllers.forEach((controller) => controller.abort());
      activeChatAbortControllers.clear();
      runStreamRecoveryControllers.forEach((controller) => controller.abort());
      runStreamRecoveryControllers.clear();
      stopRequestedConversationIds.clear();
      if (copyFeedbackTimeoutRef.current) {
        window.clearTimeout(copyFeedbackTimeoutRef.current);
      }
    };
  }, []);

  const bisqueRootForAuth = useMemo(() => {
    const preferred =
      bisqueNavLinks?.home && bisqueNavLinks.home.length > 0
        ? inferBisqueRootFromUrl(bisqueNavLinks.home)
        : null;
    if (preferred) {
      return preferred;
    }
    const fallback = inferBisqueRootFromUrl(DEFAULT_BISQUE_BROWSER_URL);
    if (fallback) {
      return fallback;
    }
    return "http://localhost:8080";
  }, [bisqueNavLinks]);

  useEffect(() => {
    let isCancelled = false;
    void apiClient
      .getBisqueSession()
      .then((session) => {
        if (isCancelled) {
          return;
        }
        const nextAuthProvider: AuthProvider =
          session.provider === "workos" || session.mode === "workos" ? "workos" : "local";
        setAuthProvider(nextAuthProvider);
        const sessionBisqueRoot = String(session.bisque_root ?? "").trim();
        if (sessionBisqueRoot) {
          setBisqueNavLinks(buildBisqueNavLinks(sessionBisqueRoot));
        }
        if (session.authenticated) {
          const sessionUser =
            String(session.username ?? "").trim() ||
            String(session.user?.email ?? "").trim() ||
            String(session.user?.username ?? "").trim() ||
            String(session.user?.id ?? "").trim() ||
            null;
          setAuthStatus("authenticated");
          setAuthUser(sessionUser);
          setAuthMode(
            session.mode === "guest"
              ? "guest"
              : session.mode === "workos"
                ? "workos"
                : "bisque"
          );
          setAuthIsAdmin(Boolean(session.is_admin));
          setBisqueCredentialsLinked(Boolean(session.bisque_linked));
          setAuthNotice(null);
          return;
        }
        setAuthStatus("unauthenticated");
        setAuthUser(null);
        setAuthMode(null);
        setAuthIsAdmin(false);
        setBisqueCredentialsLinked(false);
        setAuthNotice(accountApprovalMessageFromSession(session));
      })
      .catch(() => {
        if (isCancelled) {
          return;
        }
        setAuthProvider("local");
        setAuthStatus("unauthenticated");
        setAuthUser(null);
        setAuthMode(null);
        setAuthIsAdmin(false);
        setBisqueCredentialsLinked(false);
        setAuthNotice(null);
      });
    return () => {
      isCancelled = true;
    };
  }, [apiClient]);

  useEffect(() => {
    if (authStatus !== "authenticated") {
      return;
    }

    let cancelled = false;
    const preload = () => {
      if (cancelled) {
        return;
      }
      void preloadSecondaryPanelModules({ includeAdmin: authIsAdmin }).catch(
        () => undefined
      );
    };

    if (typeof window.requestIdleCallback === "function") {
      const idleId = window.requestIdleCallback(preload, { timeout: 2_500 });
      return () => {
        cancelled = true;
        window.cancelIdleCallback(idleId);
      };
    }

    const timeoutId = window.setTimeout(preload, 650);
    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [authIsAdmin, authStatus]);

  useEffect(() => {
    clearAuthErrorFromLocation();
  }, []);

  useEffect(() => {
    if (authStatus !== "authenticated") {
      persistedConversationHashesRef.current = {};
      optimisticConversationIdsRef.current = new Set();
      hydratingConversationIdsRef.current = new Set();
      return queueEffectUpdate(() => {
        setConversationsHydrated(false);
        setConversationListOffset(0);
        setConversationListHasMore(false);
        setConversationListLoadingMore(false);
      });
    }
    let isCancelled = false;
    const cancelQueuedReset = queueEffectUpdate(() => {
      if (!isCancelled) {
        setConversationsHydrated(false);
      }
    });
    void (async () => {
      const targetConversationId = readConversationIdFromLocation();
      try {
        const payload = await listSessionConversations(apiClient, {
          limit: CONVERSATION_PAGE_SIZE,
        });
        if (isCancelled) {
          return;
        }
        let restored = payload.conversations
          .map(conversationFromRecord)
          .sort((a, b) => b.updatedAt - a.updatedAt);
        setConversationListOffset(payload.offset + payload.count);
        setConversationListHasMore(payload.has_more);

        let missingRequestedConversationId: string | null = null;
        if (
          targetConversationId &&
          !restored.some((conversation) => conversation.id === targetConversationId)
        ) {
          try {
            const targetRecord = await apiClient.getConversation(targetConversationId);
            if (isCancelled) {
              return;
            }
            // The requested conversation may already be present under its
            // resolved id (the URL can reference it by thread id while the list
            // holds it under the local conversation id). Dedupe by resolved id
            // so we never render two history rows with the same React key.
            restored = prependResolvedConversation(conversationFromRecord(targetRecord), restored);
            setUiErrorBanner(null);
          } catch (error) {
            if (isCancelled) {
              return;
            }
            if (error instanceof ApiError && error.status === 404) {
              missingRequestedConversationId = targetConversationId;
              setUiErrorBanner(MISSING_REQUESTED_CONVERSATION_MESSAGE);
            } else {
              setUiErrorBanner(`Failed to open chat from URL: ${normalizeApiError(error)}`);
            }
          }
        }

        if (restored.length === 0) {
          const seed = createConversationState();
          optimisticConversationIdsRef.current.add(seed.id);
          setConversations([seed]);
          setActiveConversationId(seed.id);
          persistedConversationHashesRef.current = {};
          setConversationsHydrated(true);
          return;
        }
        let mergedConversations = restored;
        setConversations((current) => {
          const restoredIds = new Set(restored.map((conversation) => conversation.id));
          const optimisticLocals = current.filter(
            (conversation) =>
              optimisticConversationIdsRef.current.has(conversation.id) &&
              shouldKeepOptimisticConversationAfterHydration({
                conversationId: conversation.id,
                incomingConversationIds: restoredIds,
                missingRequestedConversationId,
              })
          );
          mergedConversations = mergeConversationPage(optimisticLocals, restored);
          return mergedConversations;
        });
        persistedConversationHashesRef.current = hashHydratedConversations(mergedConversations);
        setActiveConversationId((current) => {
          if (
            targetConversationId &&
            mergedConversations.some((conversation) => conversation.id === targetConversationId)
          ) {
            return targetConversationId;
          }
          if (current && mergedConversations.some((conversation) => conversation.id === current)) {
            return current;
          }
          return mergedConversations[0].id;
        });
        setConversationsHydrated(true);
      } catch {
        if (isCancelled) {
          return;
        }
        const seed = createConversationState();
        optimisticConversationIdsRef.current.add(seed.id);
        setConversations([seed]);
        setActiveConversationId(seed.id);
        setConversationListOffset(0);
        setConversationListHasMore(false);
        persistedConversationHashesRef.current = {};
        setConversationsHydrated(true);
      }
    })();
    return () => {
      isCancelled = true;
      cancelQueuedReset();
    };
  }, [apiClient, authStatus, hashHydratedConversations]);

  useEffect(() => {
    if (!conversationsHydrated) {
      return;
    }
    if (conversations.length === 0) {
      const seed = createConversationState();
      return queueEffectUpdate(() => {
        setConversations([seed]);
        setActiveConversationId(seed.id);
      });
    }
    if (
      !activeConversationId ||
      !conversations.some((conversation) => conversation.id === activeConversationId)
    ) {
      return queueEffectUpdate(() => {
        setActiveConversationId(conversations[0].id);
      });
    }
  }, [activeConversationId, conversations, conversationsHydrated]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !conversationsHydrated) {
      return;
    }
    const resolvedConversationId =
      activeConversationId && conversations.some((conversation) => conversation.id === activeConversationId)
        ? activeConversationId
        : conversations[0]?.id ?? null;
    replaceConversationIdInLocation(resolvedConversationId);
  }, [activeConversationId, authStatus, conversations, conversationsHydrated]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !conversationsHydrated) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      const records = conversations
        .filter(shouldPersistConversationSnapshot)
        .map(conversationToRecord);
      const previousHashes = persistedConversationHashesRef.current;
      const nextHashes: Record<string, string> = {};
      const changedRecords = records.filter((record) => {
        const fingerprint = JSON.stringify(record);
        nextHashes[record.conversation_id] = fingerprint;
        return previousHashes[record.conversation_id] !== fingerprint;
      });
      persistedConversationHashesRef.current = nextHashes;
      if (changedRecords.length === 0) {
        return;
      }
      void Promise.allSettled(
        changedRecords.map((record) => apiClient.upsertConversation(record))
      );
    }, 250);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [apiClient, authStatus, conversations, conversationsHydrated]);

  useEffect(() => {
    let isCancelled = false;
    void apiClient
      .getPublicConfig()
      .then((payload) => {
        if (isCancelled) {
          return;
        }
        if (typeof payload.bisque_guest_enabled === "boolean") {
          setAuthGuestEnabled(payload.bisque_guest_enabled);
        }
        const explicitLinks = payload.bisque_urls;
        if (explicitLinks && typeof explicitLinks === "object") {
          const home = String(explicitLinks.home ?? "").trim();
          const datasets = String(explicitLinks.datasets ?? "").trim();
          const images = String(explicitLinks.images ?? "").trim();
          const tables = String(explicitLinks.tables ?? "").trim();
          if (home && datasets && images && tables) {
            setBisqueNavLinks({ home, datasets, images, tables });
            return;
          }
        }
        const root =
          String(payload.bisque_root ?? "").trim() ||
          inferBisqueRootFromUrl(String(payload.bisque_browser_url ?? ""));
        if (root) {
          setBisqueNavLinks(buildBisqueNavLinks(root));
        }
      })
      .catch(() => {
        // non-blocking: keep UI usable if config endpoint is unavailable
      });

    return () => {
      isCancelled = true;
    };
  }, [apiClient]);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      setDebouncedResourceQuery(resourceQuery);
    }, 180);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [resourceQuery]);

  useEffect(() => {
    if (authStatus !== "authenticated") {
      resourceListKeyRef.current = "";
      return queueEffectUpdate(() => {
        setActiveResourceCollectionId(null);
        setActiveResourceCollectionSnapshot(null);
        setResources([]);
        setResourceTotalCount(0);
        setResourcesError(null);
        setResourcesLoading(false);
        setResourcesLoadingMore(false);
      });
    }
    let cancelled = false;
    const activeQuery = debouncedResourceQuery.trim();
    const activeTagFilters = parseResourceTagFilter(resourceTagFilter);
    const activeCollectionId = activeResourceCollection?.collection_id ?? "";
    const activeResourceListKey = [
      activeCollectionId ? `folder:${activeCollectionId}` : "library",
      activeQuery,
      resourceKindFilter,
      resourceSourceFilter,
      resourceSharingFilter,
      resourceStatusFilter,
      activeTagFilters.join("\u0001"),
      String(resourceRefreshToken),
    ].join("\u0000");
    resourceListKeyRef.current = activeResourceListKey;
    const cancelQueuedReset = queueEffectUpdate(() => {
      if (cancelled || resourceListKeyRef.current !== activeResourceListKey) {
        return;
      }
      setResourcesLoading(true);
      setResourcesLoadingMore(false);
      setResourcesError(null);
      setResources([]);
    });
    const request = activeCollectionId
      ? loadResourceFolderResources(apiClient, activeCollectionId, {
          limit: RESOURCE_PAGE_SIZE,
          offset: 0,
          query: activeQuery || undefined,
          kind: resourceKindFilter,
          source: resourceSourceFilter,
          sharing: resourceSharingFilter,
          status: resourceStatusFilter,
          tags: activeTagFilters,
        })
      : loadLibraryResources(apiClient, {
          limit: RESOURCE_PAGE_SIZE,
          offset: 0,
          query: activeQuery || undefined,
          kind: resourceKindFilter,
          source: resourceSourceFilter,
          sharing: resourceSharingFilter,
          status: resourceStatusFilter,
          tags: activeTagFilters,
        });
    void request
      .then((payload) => {
        if (cancelled || resourceListKeyRef.current !== activeResourceListKey) {
          return;
        }
        setResources(payload.resources);
        setResourceTotalCount(Math.max(0, Math.floor(Number(payload.count) || 0)));
      })
      .catch((error) => {
        if (cancelled || resourceListKeyRef.current !== activeResourceListKey) {
          return;
        }
        setResources([]);
        setResourceTotalCount(0);
        setResourcesError(normalizeApiError(error));
      })
      .finally(() => {
        if (cancelled || resourceListKeyRef.current !== activeResourceListKey) {
          return;
        }
        setResourcesLoading(false);
      });
    return () => {
      cancelled = true;
      cancelQueuedReset();
    };
  }, [
    apiClient,
    activeResourceCollection?.collection_id,
    authStatus,
    debouncedResourceQuery,
    resourceKindFilter,
    resourceRefreshToken,
    resourceSharingFilter,
    resourceSourceFilter,
    resourceStatusFilter,
    resourceTagFilter,
  ]);

  useEffect(() => {
    if (authStatus !== "authenticated") {
      return queueEffectUpdate(() => {
        setResourceCollections([]);
        setResourceCollectionsLoading(false);
      });
    }
    let cancelled = false;
    const cancelQueuedReset = queueEffectUpdate(() => {
      if (!cancelled) {
        setResourceCollectionsLoading(true);
      }
    });
    void loadResourceFolders(apiClient, {
      limit: 200,
      query: debouncedResourceQuery,
      status: resourceStatusFilter,
    })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setResourceCollections(payload.collections);
        setActiveResourceCollectionSnapshot((current) => {
          if (!activeResourceCollectionId) {
            return current;
          }
          return (
            payload.collections.find(
              (collection) => collection.collection_id === activeResourceCollectionId
            ) ?? current
          );
        });
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setResourceCollections([]);
        setResourcesError(normalizeApiError(error));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setResourceCollectionsLoading(false);
      });
    return () => {
      cancelled = true;
      cancelQueuedReset();
    };
  }, [
    activeResourceCollectionId,
    apiClient,
    authStatus,
    resourceCollectionRefreshToken,
    resourceStatusFilter,
    debouncedResourceQuery,
  ]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !composerResourcePickerOpen) {
      return queueEffectUpdate(() => {
        setComposerResources([]);
        setComposerResourcesError(null);
        setComposerResourcesLoading(false);
      });
    }
    let cancelled = false;
    const cancelQueuedReset = queueEffectUpdate(() => {
      if (cancelled) {
        return;
      }
      setComposerResourcesLoading(true);
      setComposerResourcesError(null);
    });
    void loadComposerResources(apiClient, {
      limit: 200,
      query: composerResourceQuery.trim() || undefined,
    })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setComposerResources(payload.resources);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setComposerResources([]);
        setComposerResourcesError(normalizeApiError(error));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setComposerResourcesLoading(false);
      });
    return () => {
      cancelled = true;
      cancelQueuedReset();
    };
  }, [apiClient, authStatus, composerResourcePickerOpen, composerResourceQuery]);

  useEffect(() => {
    return queueEffectUpdate(() => {
      setDismissedSlashPrompt(null);
      setActiveComposerResourceId(null);
      setComposerResourcePickerSelection({});
      setComposerResourceQuery("");
      setComposerResourcePickerOpen(false);
    });
  }, [activeConversationId]);

  useEffect(() => {
    if (activePanel === "admin" && !authIsAdmin) {
      return queueEffectUpdate(() => {
        setActivePanel("chat");
      });
    }
  }, [activePanel, authIsAdmin]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !authIsAdmin || activePanel !== "admin") {
      return queueEffectUpdate(() => {
        setAdminOverview(null);
        setAdminLoadingOverview(false);
      });
    }
    let cancelled = false;
    const cancelQueuedReset = queueEffectUpdate(() => {
      if (cancelled) {
        return;
      }
      setAdminLoadingOverview(true);
      setAdminError(null);
    });
    void loadAdminOverview(apiClient, { topUsers: 8, issueLimit: 12 })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setAdminOverview(payload);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setAdminError(normalizeApiError(error));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setAdminLoadingOverview(false);
      });
    return () => {
      cancelled = true;
      cancelQueuedReset();
    };
  }, [activePanel, adminRefreshToken, apiClient, authIsAdmin, authStatus]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !authIsAdmin || activePanel !== "admin") {
      return queueEffectUpdate(() => {
        setAdminOrganizations([]);
        setAdminLoadingOrganizations(false);
      });
    }
    let cancelled = false;
    const cancelQueuedReset = queueEffectUpdate(() => {
      if (cancelled) {
        return;
      }
      setAdminLoadingOrganizations(true);
    });
    void loadAdminOrganizations(apiClient, { limit: 250 })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setAdminOrganizations(payload.organizations);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setAdminError(normalizeApiError(error));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setAdminLoadingOrganizations(false);
      });
    return () => {
      cancelled = true;
      cancelQueuedReset();
    };
  }, [activePanel, adminRefreshToken, apiClient, authIsAdmin, authStatus]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !authIsAdmin || activePanel !== "admin") {
      return queueEffectUpdate(() => {
        setAdminUsers([]);
        setAdminLoadingUsers(false);
      });
    }
    let cancelled = false;
    const cancelQueuedReset = queueEffectUpdate(() => {
      if (cancelled) {
        return;
      }
      setAdminLoadingUsers(true);
    });
    void loadAdminUsers(apiClient, {
      limit: 250,
      query: adminUserQuery.trim() || undefined,
    })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setAdminUsers(payload.users);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setAdminError(normalizeApiError(error));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setAdminLoadingUsers(false);
      });
    return () => {
      cancelled = true;
      cancelQueuedReset();
    };
  }, [activePanel, adminRefreshToken, adminUserQuery, apiClient, authIsAdmin, authStatus]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !authIsAdmin || activePanel !== "admin") {
      return queueEffectUpdate(() => {
        setAdminRuns([]);
        setAdminLoadingRuns(false);
      });
    }
    let cancelled = false;
    const cancelQueuedReset = queueEffectUpdate(() => {
      if (cancelled) {
        return;
      }
      setAdminLoadingRuns(true);
    });
    void loadAdminRuns(apiClient, {
      limit: 250,
      status: adminRunStatusFilter || undefined,
      query: adminRunQuery.trim() || undefined,
    })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setAdminRuns(payload.runs);
        setActiveAdminRunEventRunId((current) =>
          current && payload.runs.some((run) => run.run_id === current) ? current : null
        );
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setAdminError(normalizeApiError(error));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setAdminLoadingRuns(false);
      });
    return () => {
      cancelled = true;
      cancelQueuedReset();
    };
  }, [
    activePanel,
    adminRefreshToken,
    adminRunQuery,
    adminRunStatusFilter,
    apiClient,
    authIsAdmin,
    authStatus,
  ]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !authIsAdmin || activePanel !== "admin") {
      return queueEffectUpdate(() => {
        setAdminIssues([]);
        setAdminLoadingIssues(false);
      });
    }
    let cancelled = false;
    const cancelQueuedReset = queueEffectUpdate(() => {
      if (cancelled) {
        return;
      }
      setAdminLoadingIssues(true);
    });
    void loadAdminIssues(apiClient, 25)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setAdminIssues(payload.issues);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        setAdminError(normalizeApiError(error));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setAdminLoadingIssues(false);
      });
    return () => {
      cancelled = true;
      cancelQueuedReset();
    };
  }, [activePanel, adminRefreshToken, apiClient, authIsAdmin, authStatus]);

  const activeConversation = useMemo(() => {
    if (conversations.length === 0) {
      return null;
    }
    if (!activeConversationId) {
      return conversations[0];
    }
    return (
      conversations.find((conversation) => conversation.id === activeConversationId) ??
      conversations[0]
    );
  }, [activeConversationId, conversations]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !conversationsHydrated) {
      return;
    }
    if (!activeConversation || activeConversation.hydrated) {
      return;
    }
    return queueEffectUpdate(() => {
      void ensureConversationHydrated(activeConversation.id);
    });
  }, [
    activeConversation,
    authStatus,
    conversationsHydrated,
    ensureConversationHydrated,
  ]);

  const updateConversation = useCallback((
    conversationId: string,
    updater: (conversation: ConversationState) => ConversationState
  ): void => {
    setConversations((previous) =>
      previous.map((conversation) =>
        conversation.id === conversationId ? updater(conversation) : conversation
      )
    );
  }, []);

  const applyGeneratedConversationTitle = useCallback(
    (
      conversationId: string,
      response: ChatResponse | null | undefined,
      temporaryTitle?: string
    ): void => {
      const generatedTitle = generatedConversationTitleFromResponse(response);
      if (!generatedTitle) {
        return;
      }
      updateConversation(conversationId, (conversation) => {
        if (
          !conversation.messages.some((message) => message.role === "user") ||
          !shouldApplyGeneratedConversationTitle(
            conversation.title,
            generatedTitle,
            temporaryTitle
          )
        ) {
          return conversation;
        }
        return {
          ...conversation,
          title: generatedTitle,
          updatedAt: Date.now(),
        };
      });
    },
    [updateConversation]
  );

  const updateActiveConversation = useCallback((
    updater: (conversation: ConversationState) => ConversationState
  ): void => {
    if (!activeConversation) {
      return;
    }
    updateConversation(activeConversation.id, updater);
  }, [activeConversation, updateConversation]);

  const clearComposerDraft = useCallback((conversationId: string): void => {
    const normalizedConversationId = String(conversationId || "").trim();
    if (!normalizedConversationId) {
      return;
    }
    setComposerDraftsByConversationId((previous) => {
      if (!Object.prototype.hasOwnProperty.call(previous, normalizedConversationId)) {
        return previous;
      }
      const next = { ...previous };
      delete next[normalizedConversationId];
      return next;
    });
  }, []);

  const focusComposerTextarea = useCallback((): void => {
    const textarea = composerTextareaRef.current;
    if (!textarea) {
      return;
    }
    window.requestAnimationFrame(() => {
      textarea.focus();
      const selectionEnd = textarea.value.length;
      textarea.setSelectionRange(selectionEnd, selectionEnd);
    });
  }, []);

  const rememberActiveConversationScrollPosition = useCallback((): void => {
    const conversationId = activeConversation?.id ?? null;
    const scrollElement = activeChatScrollElementRef.current;
    if (!conversationId || !scrollElement) {
      return;
    }
    conversationScrollMemoryRef.current[conversationId] =
      captureConversationScrollMemory(scrollElement);
    conversationScrollWriteBlockRef.current = conversationId;
  }, [activeConversation?.id]);

  const clearActiveComposerWorkflowPreset = useCallback((): void => {
    if (!activeConversation) {
      return;
    }
    updateConversation(activeConversation.id, (conversation) => {
      if (!conversation.composerWorkflowPreset) {
        return conversation;
      }
      return {
        ...conversation,
        updatedAt: Date.now(),
        composerWorkflowPreset: null,
      };
    });
  }, [activeConversation, updateConversation]);

  const openComposerResourcePicker = useCallback(
    ({ clearSelection = true }: { clearSelection?: boolean } = {}): void => {
      setActivePanel("chat");
      setResourceViewerContext(null);
      setComposerResourcePickerOpen(true);
      setActiveComposerResourceId(null);
      setComposerResourceQuery("");
      if (clearSelection) {
        setComposerResourcePickerSelection({});
      }
    },
    []
  );

  const createNewConversation = useCallback((): void => {
    const reusableBlankDraft = findReusableBlankDraftConversation(
      conversations,
      activeConversation?.id ?? activeConversationId
    );
    if (reusableBlankDraft) {
      rememberActiveConversationScrollPosition();
      flushSync(() => {
        setActiveConversationId(reusableBlankDraft.id);
        setActivePanel("chat");
        setViewerOpen(false);
        setResourceViewerContext(null);
        setComposerResourcePickerOpen(false);
        setComposerResourcePickerSelection({});
        setActiveComposerResourceId(null);
        setComposerResourceQuery("");
        setDismissedSlashPrompt(null);
        setUiErrorBanner(null);
      });
      return;
    }

    const nextConversation = createConversationState();
    optimisticConversationIdsRef.current.add(nextConversation.id);
    rememberActiveConversationScrollPosition();
    flushSync(() => {
      setConversations((previous) => [nextConversation, ...previous]);
      setActiveConversationId(nextConversation.id);
      setActivePanel("chat");
      setViewerOpen(false);
      setResourceViewerContext(null);
      setComposerResourcePickerOpen(false);
      setComposerResourcePickerSelection({});
      setActiveComposerResourceId(null);
      setComposerResourceQuery("");
      setDismissedSlashPrompt(null);
      setUiErrorBanner(null);
    });
  }, [
    activeConversation?.id,
    activeConversationId,
    conversations,
    rememberActiveConversationScrollPosition,
  ]);

  const openResourcesPanel = useCallback((): void => {
    rememberActiveConversationScrollPosition();
    setActivePanel("resources");
    setViewerOpen(false);
    setResourceViewerContext(null);
    setResourceRefreshToken((value) => value + 1);
  }, [rememberActiveConversationScrollPosition]);

  const openTrainingPanel = useCallback((): void => {
    rememberActiveConversationScrollPosition();
    setActivePanel("training");
    setViewerOpen(false);
    setResourceViewerContext(null);
    setResourceRefreshToken((value) => value + 1);
  }, [rememberActiveConversationScrollPosition]);

  const openScientificViewerPanel = useCallback((): void => {
    rememberActiveConversationScrollPosition();
    setActivePanel("scientific-viewer");
    setViewerOpen(false);
    setResourceViewerContext(null);
    setResourceRefreshToken((value) => value + 1);
  }, [rememberActiveConversationScrollPosition]);

  const openBisqueHome = useCallback((): void => {
    const homeUrl = String(bisqueNavLinks?.home ?? "").trim();
    if (!homeUrl || typeof window === "undefined") {
      return;
    }
    window.open(homeUrl, "_blank", "noopener,noreferrer");
  }, [bisqueNavLinks?.home]);

  useEffect(() => {
    if (authStatus !== "authenticated") {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent): void => {
      const usesCommonModifiers =
        (event.metaKey || event.ctrlKey) &&
        event.shiftKey &&
        !event.altKey;
      if (!usesCommonModifiers || event.defaultPrevented || event.isComposing) {
        return;
      }
      const key = event.key.toLowerCase();
      const shortcutAction =
        key === NEW_CHAT_SHORTCUT_KEY
          ? createNewConversation
          : key === RESOURCES_SHORTCUT_KEY
            ? openResourcesPanel
            : key === TRAINING_SHORTCUT_KEY
              ? openTrainingPanel
            : key === GO_TO_BISQUE_SHORTCUT_KEY
              ? openBisqueHome
              : null;
      if (!shortcutAction) {
        return;
      }
      if (isEditableEventTarget(event.target)) {
        return;
      }
      event.preventDefault();
      shortcutAction();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    authStatus,
    createNewConversation,
    openBisqueHome,
    openResourcesPanel,
    openTrainingPanel,
  ]);

  const isChatStopRequested = useCallback((conversationId: string): boolean => {
    const normalizedConversationId = conversationId.trim();
    return (
      normalizedConversationId.length > 0 &&
      stopRequestedConversationIdsRef.current.has(normalizedConversationId)
    );
  }, []);

  const finalizeStoppedConversation = useCallback(
    ({
      conversationId,
      assistantMessageId,
      streamedText,
    }: {
      conversationId: string;
      assistantMessageId?: string | null;
      streamedText: string;
    }): void => {
      const partialText = streamedText.trim();
      updateConversation(conversationId, (conversation) => {
        const messages = assistantMessageId
          ? conversation.messages.flatMap((message) => {
              if (message.id !== assistantMessageId) {
                return [message];
              }
              const preservedText = partialText || message.content.trim();
              if (!preservedText) {
                return [];
              }
              return [
                {
                  ...message,
                  content: preservedText,
                  liveStream: undefined,
                },
              ];
            })
          : conversation.messages;
        return {
          ...conversation,
          updatedAt: Date.now(),
          sending: false,
          chatError: null,
          streamingMessageId:
            assistantMessageId && conversation.streamingMessageId === assistantMessageId
              ? null
              : conversation.streamingMessageId,
          messages,
        };
      });
    },
    [updateConversation]
  );

  const requestStopConversation = useCallback(
    (conversationId: string): void => {
      const normalizedConversationId = conversationId.trim();
      if (!normalizedConversationId) {
        return;
      }
      stopRequestedConversationIdsRef.current.add(normalizedConversationId);
      if (reuseDecisionResolverRef.current) {
        reuseDecisionResolverRef.current("rerun");
        reuseDecisionResolverRef.current = null;
      }
      setPendingReusePrompt(null);
      const controller = activeChatAbortControllersRef.current.get(normalizedConversationId);
      if (controller && !controller.signal.aborted) {
        controller.abort();
      }
    },
    [setPendingReusePrompt]
  );

  const deleteConversationFromHistory = async (
    conversationId: string
  ): Promise<void> => {
    const target = conversations.find((item) => item.id === conversationId);
    if (!target) {
      return;
    }
    setConversationDeletingById((previous) => ({
      ...previous,
      [conversationId]: true,
    }));
    try {
      await apiClient.deleteConversation(conversationId);
      setUiErrorBanner(null);
      clearComposerDraft(conversationId);
      setConversations((previous) => {
        const filtered = previous.filter((item) => item.id !== conversationId);
        if (filtered.length === 0) {
          const seed = createConversationState();
          setActiveConversationId(seed.id);
          return [seed];
        }
        setActiveConversationId((current) => {
          if (
            !current ||
            current === conversationId ||
            !filtered.some((item) => item.id === current)
          ) {
            return filtered[0].id;
          }
          return current;
        });
        return filtered;
      });
      setViewerOpen(false);
      setResourceViewerContext(null);
    } catch (error) {
      setUiErrorBanner(`Failed to delete conversation: ${normalizeApiError(error)}`);
    } finally {
      setConversationDeletingById((previous) => {
        const next = { ...previous };
        delete next[conversationId];
        return next;
      });
    }
  };

  const requestConversationDelete = (conversationId: string): void => {
    const target = conversations.find((item) => item.id === conversationId);
    if (!target) {
      return;
    }
    setPendingConversationRename((current) =>
      current?.id === conversationId ? null : current
    );
    setPendingConversationDelete({ id: target.id, title: target.title });
  };

  const startConversationRename = useCallback(
    (conversationId: string, conversationTitle: string): void => {
      if (conversationDeletingById[conversationId] || conversationRenamingById[conversationId]) {
        return;
      }
      setPendingConversationDelete(null);
      setPendingConversationRename({
        id: conversationId,
        title: normalizeConversationTitle(conversationTitle),
      });
    },
    [conversationDeletingById, conversationRenamingById]
  );

  const cancelConversationRename = useCallback((): void => {
    setPendingConversationRename(null);
  }, []);

  const updatePendingConversationRenameTitle = useCallback(
    (conversationId: string, nextTitle: string): void => {
      setPendingConversationRename((current) =>
        current?.id === conversationId ? { ...current, title: nextTitle } : current
      );
    },
    []
  );

  const submitConversationRename = useCallback(async (): Promise<void> => {
    if (!pendingConversationRename) {
      return;
    }
    const conversationId = pendingConversationRename.id;
    const nextTitle = normalizeConversationTitle(pendingConversationRename.title);
    const currentConversation = conversations.find((item) => item.id === conversationId);
    if (!currentConversation) {
      setPendingConversationRename(null);
      return;
    }
    if (nextTitle === normalizeConversationTitle(currentConversation.title)) {
      setPendingConversationRename(null);
      return;
    }
    setConversationRenamingById((previous) => ({
      ...previous,
      [conversationId]: true,
    }));
    try {
      let sourceConversation = currentConversation;
      if (!sourceConversation.hydrated) {
        const record = await apiClient.getConversation(conversationId);
        sourceConversation = conversationFromRecord(record);
      }
      const renamedConversation: ConversationState = {
        ...sourceConversation,
        title: nextTitle,
        updatedAt: Date.now(),
      };
      const savedRecord = await apiClient.upsertConversation(
        conversationToRecord(renamedConversation),
        { titleSource: "manual" }
      );
      const savedConversation = conversationFromRecord(savedRecord);
      setConversations((previous) =>
        mergeConversationPage(previous, [savedConversation])
      );
      persistedConversationHashesRef.current = {
        ...persistedConversationHashesRef.current,
        [conversationId]: JSON.stringify(conversationToRecord(savedConversation)),
      };
      setUiErrorBanner(null);
      setPendingConversationRename(null);
    } catch (error) {
      setUiErrorBanner(`Failed to rename conversation: ${normalizeApiError(error)}`);
    } finally {
      setConversationRenamingById((previous) => {
        const next = { ...previous };
        delete next[conversationId];
        return next;
      });
    }
  }, [apiClient, conversations, pendingConversationRename]);

  const authenticateBisque = async (payload: {
    username: string;
    password: string;
  }): Promise<void> => {
    setAuthSubmitting(true);
    setAuthError(null);
    setAuthNotice(null);
    try {
      const session = await apiClient.loginBisque(payload);
      if (!session.authenticated) {
        throw new Error("Authentication did not complete.");
      }
      setAuthUser(String(session.username ?? payload.username).trim() || payload.username);
      setAuthMode(session.mode === "workos" ? "workos" : "bisque");
      setAuthProvider(
        session.provider === "workos" || session.mode === "workos" ? "workos" : "local"
      );
      setAuthIsAdmin(Boolean(session.is_admin));
      setBisqueCredentialsLinked(Boolean(session.bisque_linked));
      setAuthStatus("authenticated");
      setAuthNotice(null);
    } catch (error) {
      setAuthStatus("unauthenticated");
      setAuthUser(null);
      setAuthMode(null);
      setAuthIsAdmin(false);
      setBisqueCredentialsLinked(false);
      setAuthError(normalizeApiError(error));
      throw error;
    } finally {
      setAuthSubmitting(false);
    }
  };

  const startHostedAuth = useCallback(async (): Promise<void> => {
    setAuthSubmitting(true);
    setAuthError(null);
    setAuthNotice(null);
    try {
      const session = await apiClient.startHostedAuth();
      const authorizationUrl = String(session.authorization_url ?? "").trim();
      if (!authorizationUrl) {
        throw new Error("Hosted sign-in did not return an authorization URL.");
      }
      if (typeof window !== "undefined") {
        window.location.assign(authorizationUrl);
      }
    } catch (error) {
      setAuthStatus("unauthenticated");
      setAuthUser(null);
      setAuthMode(null);
      setAuthIsAdmin(false);
      setBisqueCredentialsLinked(false);
      setAuthError(normalizeApiError(error));
      throw error;
    } finally {
      setAuthSubmitting(false);
    }
  }, [apiClient]);

  const retryHostedAuth = useCallback(async (): Promise<void> => {
    hostedAuthRedirectAttemptedRef.current = false;
    await startHostedAuth();
  }, [startHostedAuth]);

  useEffect(() => {
    if (authProvider !== "workos") {
      hostedAuthRedirectAttemptedRef.current = false;
      return;
    }
    if (authStatus !== "unauthenticated") {
      return;
    }
    // A pending/disabled account notice means WorkOS sign-in succeeded but the
    // Ultra account is not approved; auto-redirecting again would loop through
    // AuthKit forever.
    if (authError || authNotice || hostedAuthRedirectAttemptedRef.current) {
      return;
    }
    hostedAuthRedirectAttemptedRef.current = true;
    void startHostedAuth().catch(() => {
      // startHostedAuth stores the visible auth error state.
    });
  }, [authError, authNotice, authProvider, authStatus, startHostedAuth]);

  const loadCurrentUserProfile = useCallback(
    () => apiClient.getCurrentUser(),
    [apiClient]
  );
  const saveCurrentUserProfile = useCallback(
    (profile: CurrentUserProfile) => apiClient.updateCurrentUser(profile),
    [apiClient]
  );
  const loadCurrentUserTokenUsage = useCallback(
    (days: number) => apiClient.getTokenUsage(days),
    [apiClient]
  );

  const linkBisqueAccountFromSettings = useCallback(
    async (payload: { username: string; password: string }): Promise<{ imageCount: number }> => {
      const session = await apiClient.loginBisque(payload);
      if (!session.authenticated) {
        throw new Error("Authentication did not complete.");
      }
      if (!session.bisque_linked) {
        throw new Error("BisQue account link did not return a credential-backed session.");
      }
      const sessionBisqueRoot = String(session.bisque_root ?? "").trim();
      if (sessionBisqueRoot) {
        setBisqueNavLinks(buildBisqueNavLinks(sessionBisqueRoot));
      }
      setAuthUser(String(session.username ?? payload.username).trim() || payload.username);
      setAuthMode(session.mode === "workos" ? "workos" : "bisque");
      setAuthProvider(
        session.provider === "workos" || session.mode === "workos" ? "workos" : "local"
      );
      setAuthIsAdmin(Boolean(session.is_admin));
      setBisqueCredentialsLinked(Boolean(session.bisque_linked));
      setAuthStatus("authenticated");
      setAuthError(null);

      const search = await apiClient.searchBisqueResources({
        resourceType: "image",
        scope: "owner",
        limit: 1,
        countAll: true,
      });
      const imageCount = Math.max(0, Number(search.count) || 0);
      showSuccessToast("Successfully linked BisQue account", {
        description: `Found ${imageCount.toLocaleString()} image${
          imageCount === 1 ? "" : "s"
        } on BisQue.`,
      });
      return { imageCount };
    },
    [apiClient]
  );

  const requestAccount = async (payload: {
    name: string;
    email: string;
    affiliation: string;
  }): Promise<void> => {
    if (!authGuestEnabled) {
      const message = "Account requests are disabled. Sign in with your BisQue username and password.";
      setAuthError(message);
      throw new Error(message);
    }
    setAuthSubmitting(true);
    setAuthError(null);
    setAuthNotice(null);
    try {
      const session = await apiClient.requestAccount(payload);
      if (session.authenticated) {
        throw new Error("Account request unexpectedly returned an authenticated session.");
      }
      setAuthUser(null);
      setAuthMode(null);
      setAuthIsAdmin(false);
      setBisqueCredentialsLinked(false);
      setAuthStatus("unauthenticated");
      setAuthNotice(
        accountApprovalMessageFromSession(session) ||
          "Your account request is pending administrator approval."
      );
    } catch (error) {
      setAuthStatus("unauthenticated");
      setAuthUser(null);
      setAuthMode(null);
      setAuthIsAdmin(false);
      setBisqueCredentialsLinked(false);
      setAuthError(normalizeApiError(error));
      setAuthNotice(null);
      throw error;
    } finally {
      setAuthSubmitting(false);
    }
  };

  const clearAuthViewState = useCallback((): void => {
    setAuthStatus("unauthenticated");
    setAuthUser(null);
    setAuthMode(null);
    setAuthIsAdmin(false);
    setBisqueCredentialsLinked(false);
    setAuthError(null);
    setAuthNotice(null);
    setBisqueResourceCountsState({ requestKey: "", counts: null });
    setComposerDraftsByConversationId({});
    setConversations([]);
    setActiveConversationId(null);
    setConversationsHydrated(false);
    persistedConversationHashesRef.current = {};
    setActivePanel("chat");
    setViewerOpen(false);
    setResourceViewerContext(null);
    setResources([]);
    setResourcesLoading(false);
    setResourcesError(null);
    setConversationDeletingById({});
    setResourceDeletingById({});
    setAdminOverview(null);
    setAdminOrganizations([]);
    setAdminUsers([]);
    setAdminRuns([]);
    setAdminIssues([]);
    setAdminError(null);
    setAdminRunCancellingById({});
    setAdminRunRequeueingById({});
    setAdminDeletingConversationKey(null);
  }, []);

  // Re-validate the session when the user returns to a long-idle tab so an
  // expired or revoked WorkOS session routes back to sign-in instead of
  // surfacing as scattered request failures. Only a definitive
  // "authenticated: false" signs the user out; network errors are ignored.
  useEffect(() => {
    if (authStatus !== "authenticated" || typeof document === "undefined") {
      return;
    }
    const revalidateSession = () => {
      if (document.visibilityState !== "visible") {
        return;
      }
      const now = Date.now();
      if (now - sessionRevalidatedAtRef.current < 60_000) {
        return;
      }
      sessionRevalidatedAtRef.current = now;
      void apiClient
        .getBisqueSession()
        .then((session) => {
          if (session.authenticated) {
            return;
          }
          clearAuthViewState();
          setAuthNotice(accountApprovalMessageFromSession(session));
        })
        .catch(() => undefined);
    };
    document.addEventListener("visibilitychange", revalidateSession);
    return () => {
      document.removeEventListener("visibilitychange", revalidateSession);
    };
  }, [apiClient, authStatus, clearAuthViewState]);

  const logoutBisque = useCallback(async (): Promise<void> => {
    let logoutUrl = "";
    try {
      const session = await apiClient.logoutBisque();
      logoutUrl = String(session.logout_url ?? "").trim();
    } catch {
      // If logout endpoint fails, still clear local auth view state.
    }
    clearAuthViewState();
    if (logoutUrl && typeof window !== "undefined") {
      window.location.assign(logoutUrl);
    }
  }, [apiClient, clearAuthViewState]);

  const unlinkBisqueAccount = useCallback(async (): Promise<void> => {
    await apiClient.unlinkBisqueAccount();
    clearAuthViewState();
  }, [apiClient, clearAuthViewState]);

  const promptBisqueAuthentication = useCallback(async (message: string): Promise<void> => {
    const nextMessage = message.trim() || "BisQue authentication is required.";
    try {
      await apiClient.logoutBisque();
    } catch {
      // If logout fails, still move the user back to the local auth screen.
    }
    clearAuthViewState();
    setAuthError(nextMessage);
  }, [apiClient, clearAuthViewState]);

  const copyTextWithUiFeedback = useCallback(async (
    value: string,
    label: string
  ): Promise<void> => {
    const normalizedValue = value.trim();
    if (!normalizedValue) {
      return;
    }
    if (typeof navigator === "undefined" || !navigator.clipboard?.writeText) {
      setUiErrorBanner("Clipboard access is unavailable in this browser.");
      return;
    }
    try {
      await navigator.clipboard.writeText(normalizedValue);
      setUiErrorBanner(null);
    } catch (error) {
      setUiErrorBanner(`Failed to copy ${label}: ${normalizeApiError(error)}`);
    }
  }, []);

  const copyBisqueResourceUri = useCallback(async (resourceUrl: string): Promise<void> => {
    await copyTextWithUiFeedback(resourceUrl, "BisQue link");
  }, [copyTextWithUiFeedback]);

  const copyConversationLink = useCallback(async (conversationId: string): Promise<void> => {
    await copyTextWithUiFeedback(buildConversationUrl(conversationId), "chat link");
  }, [copyTextWithUiFeedback]);

  const copyConversationId = useCallback(async (conversationId: string): Promise<void> => {
    await copyTextWithUiFeedback(conversationId, "chat ID");
  }, [copyTextWithUiFeedback]);

  const isSuccessfulBisqueImportStatus = useCallback((
    status: "imported" | "reused" | "error" | string | null | undefined
  ): boolean => {
    const normalized = String(status ?? "").trim().toLowerCase();
    return normalized === "imported" || normalized === "reused";
  }, []);

  const importBisqueResourcesIntoConversation = useCallback(async (
    resourcesToImport: string[],
    options?: {
      materialize?: boolean;
      persistSelectionContext?: boolean;
      source?: string;
      suggestedDomain?: string | null;
      suggestedToolNames?: string[];
      originatingMessageId?: string | null;
      originatingUserText?: string | null;
    }
  ): Promise<BisqueImportedSelection> => {
    const conversation = activeConversation;
    if (!conversation) {
      return { uploadedFiles: [], bisqueLinksByFileId: {} };
    }
    const conversationId = conversation.id;
    const normalizedResources = resourcesToImport
      .map((item) => String(item ?? "").trim())
      .filter((item) => item.length > 0);
    if (normalizedResources.length === 0) {
      return { uploadedFiles: [], bisqueLinksByFileId: {} };
    }
    const partitionedSelectionUris = partitionBisqueUris(normalizedResources);
    const shouldMaterialize = options?.materialize !== false;
    if (options?.persistSelectionContext) {
      updateConversation(conversationId, (current) => ({
        ...current,
        updatedAt: Date.now(),
        selectionImportPending: shouldMaterialize,
        chatError: null,
      }));
    }

    const existingFileByResourceUri = new Map<string, UploadedFileRecord>();
    const existingLinkByResourceUri = new Map<string, BisqueViewerLink>();
    Object.entries(conversation.bisqueLinksByFileId).forEach(([fileId, link]) => {
      const resourceUri = String(link.resourceUri ?? "").trim();
      if (!resourceUri) {
        return;
      }
      const uploaded = conversation.uploadedFiles.find((file) => file.file_id === fileId);
      if (!uploaded) {
        return;
      }
      existingFileByResourceUri.set(resourceUri.toLowerCase(), uploaded);
      existingLinkByResourceUri.set(resourceUri.toLowerCase(), link);
    });

    const resourcesMissingImport = normalizedResources.filter(
      (resourceUri) => !existingFileByResourceUri.has(resourceUri.toLowerCase())
    );

    if (!shouldMaterialize) {
      const orderedExistingFileIds = normalizedResources
        .map((resourceUri) => existingFileByResourceUri.get(resourceUri.toLowerCase())?.file_id ?? null)
        .filter((fileId): fileId is string => Boolean(fileId));
      updateConversation(conversationId, (current) => ({
        ...current,
        updatedAt: Date.now(),
        activeSelectionContext:
          options?.persistSelectionContext
            ? ({
                context_id: makeId(),
                source: options?.source ?? "use_in_chat",
                focused_file_ids: orderedExistingFileIds,
                resource_uris: partitionedSelectionUris.resourceUris,
                dataset_uris: partitionedSelectionUris.datasetUris,
                originating_message_id: options?.originatingMessageId ?? null,
                originating_user_text:
                  options?.originatingUserText?.trim() ||
                  [...current.messages]
                    .reverse()
                    .find((message) => message.role === "user" && message.content.trim().length > 0)
                    ?.content?.trim() ||
                  null,
                suggested_domain: options?.suggestedDomain ?? null,
                suggested_tool_names: Array.from(
                  new Set((options?.suggestedToolNames ?? []).map((name) => String(name || "").trim()))
                ).filter((name) => name.length > 0),
              } satisfies SelectionContext)
            : current.activeSelectionContext,
        selectionImportPending: false,
        chatError: null,
      }));
      return { uploadedFiles: [], bisqueLinksByFileId: {} };
    }

    try {
      const importResponse =
        resourcesMissingImport.length > 0
          ? await apiClient.importBisqueResources(resourcesMissingImport)
          : { uploaded: [], imports: [], file_count: 0 };
      const importedBisqueLinks: Record<string, BisqueViewerLink> = {};
      const importedFileByResourceUri = new Map<string, UploadedFileRecord>();
      const importedLinkByResourceUri = new Map<string, BisqueViewerLink>();
      importResponse.imports.forEach((item) => {
        const fileId = item.uploaded?.file_id;
        const clientViewUrl = item.client_view_url;
        const resourceUri = String(item.resource_uri ?? "").trim();
        if (
          !isSuccessfulBisqueImportStatus(item.status) ||
          !fileId ||
          !clientViewUrl ||
          !clientViewUrl.trim() ||
          !resourceUri
        ) {
          return;
        }
        const link = {
          clientViewUrl,
          resourceUri: item.resource_uri ?? null,
          imageServiceUrl: item.image_service_url ?? null,
          inputUrl: item.input_url,
        } satisfies BisqueViewerLink;
        importedBisqueLinks[fileId] = link;
        importedLinkByResourceUri.set(resourceUri.toLowerCase(), link);
        if (item.uploaded) {
          importedFileByResourceUri.set(resourceUri.toLowerCase(), item.uploaded);
        }
      });
      updateConversation(conversationId, (current) => {
        const importedUploadedFiles = uniqueByFileId([
          ...current.uploadedFiles,
          ...importResponse.uploaded,
        ]);
        const orderedExistingFileIds = normalizedResources
          .map((resourceUri) => {
            const normalized = resourceUri.toLowerCase();
            const existing =
              existingFileByResourceUri.get(normalized) ?? importedFileByResourceUri.get(normalized);
            return existing?.file_id ?? null;
          })
          .filter((fileId): fileId is string => Boolean(fileId));
        const retainedFailedPreviews: Record<string, true> = {};
        const mergedBisqueLinks: Record<string, BisqueViewerLink> = {
          ...current.bisqueLinksByFileId,
          ...importedBisqueLinks,
        };
        const retainedBisqueLinks: Record<string, BisqueViewerLink> = {};
        importedUploadedFiles.forEach((file) => {
          if (current.failedUploadPreviewIds[file.file_id]) {
            retainedFailedPreviews[file.file_id] = true;
          }
          if (mergedBisqueLinks[file.file_id]) {
            retainedBisqueLinks[file.file_id] = mergedBisqueLinks[file.file_id];
          }
        });
        return {
          ...current,
          updatedAt: Date.now(),
          uploadedFiles: importedUploadedFiles,
          stagedUploadFileIds: uniqueFileIds([
            ...current.stagedUploadFileIds.filter((fileId) =>
              importedUploadedFiles.some((file) => file.file_id === fileId)
              ),
            ...orderedExistingFileIds,
          ]),
          activeSelectionContext:
            options?.persistSelectionContext
              ? ({
                  context_id: makeId(),
                  source: options?.source ?? "use_in_chat",
                  focused_file_ids: orderedExistingFileIds,
                  resource_uris: partitionedSelectionUris.resourceUris,
                  dataset_uris: partitionedSelectionUris.datasetUris,
                  originating_message_id: options?.originatingMessageId ?? null,
                  originating_user_text:
                    options?.originatingUserText?.trim() ||
                    [...current.messages]
                      .reverse()
                      .find(
                        (message) => message.role === "user" && message.content.trim().length > 0
                      )
                      ?.content?.trim() ||
                    null,
                  suggested_domain: options?.suggestedDomain ?? null,
                  suggested_tool_names: Array.from(
                    new Set((options?.suggestedToolNames ?? []).map((name) => String(name || "").trim()))
                  ).filter((name) => name.length > 0),
                } satisfies SelectionContext)
              : current.activeSelectionContext,
          failedUploadPreviewIds: retainedFailedPreviews,
          bisqueLinksByFileId: retainedBisqueLinks,
          selectionImportPending: false,
          chatError: null,
        };
      });
      const orderedUploads = normalizedResources
        .map((resourceUri) => {
          const normalized = resourceUri.toLowerCase();
          return (
            existingFileByResourceUri.get(normalized) ?? importedFileByResourceUri.get(normalized) ?? null
          );
        })
        .filter((file): file is UploadedFileRecord => file !== null);
      const orderedLinks = Object.fromEntries(
        normalizedResources
          .map((resourceUri) => {
            const normalized = resourceUri.toLowerCase();
            const file =
              existingFileByResourceUri.get(normalized) ?? importedFileByResourceUri.get(normalized);
            const link =
              existingLinkByResourceUri.get(normalized) ?? importedLinkByResourceUri.get(normalized);
            if (!file || !link) {
              return null;
            }
            return [file.file_id, link] as const;
          })
          .filter((entry): entry is readonly [string, BisqueViewerLink] => entry !== null)
      );
      if (orderedUploads.length > 0) {
        setActivePanel("chat");
        setResourceViewerContext(null);
        setViewerOpen(false);
        setUiErrorBanner(null);
        return {
          uploadedFiles: orderedUploads,
          bisqueLinksByFileId: orderedLinks,
        };
      }
      const failedImports = importResponse.imports.filter((item) => item.status === "error");
      if (failedImports.length > 0) {
        const details = failedImports
          .slice(0, 2)
          .map((item) => {
            const detail = item.error?.trim();
            return detail ? `${item.input_url} (${detail})` : `${item.input_url} (import failed)`;
          })
          .join("; ");
        setUiErrorBanner(`BisQue import failed: ${details}`);
      }
      return { uploadedFiles: [], bisqueLinksByFileId: {} };
    } catch (error) {
      if (options?.persistSelectionContext) {
        updateConversation(conversationId, (current) => ({
          ...current,
          updatedAt: Date.now(),
          selectionImportPending: false,
        }));
      }
      if (isBisqueAuthApiError(error)) {
        void promptBisqueAuthentication(normalizeApiError(error));
      }
      setUiErrorBanner(`BisQue import failed: ${normalizeApiError(error)}`);
      return { uploadedFiles: [], bisqueLinksByFileId: {} };
    }
  }, [
    activeConversation,
    apiClient,
    isSuccessfulBisqueImportStatus,
    promptBisqueAuthentication,
    updateConversation,
  ]);

  const resolveBisqueReferenceSelectionForPrompt = async (
    promptText: string,
    conversation: ConversationState
  ): Promise<{
    promptForModel: string;
    selectedUploads: UploadedFileRecord[];
    selectedFileIds: string[];
    quickPreviewFileIds: string[];
    resolvedRows: ToolResourceRow[];
    selectedToolNames: string[];
    selectionContext: SelectionContext | null;
  } | null> => {
    const selection = inferBisqueReferenceSelection(promptText, conversation.messages);
    if (!selection) {
      return null;
    }
    const resourceUris = selection.selectedRows
      .map((row) => String(row.resourceUri ?? "").trim())
      .filter(Boolean);
    if (resourceUris.length === 0) {
      return null;
    }
    const selectedToolNames = inferBisqueSelectionToolNames(promptText, {
      hasSelectionContext: true,
      hasStagedUploads: false,
    });
    const importedSelection = await importBisqueResourcesIntoConversation(resourceUris, {
      materialize: selectedToolNames.length === 0,
      persistSelectionContext: true,
      source: "deictic_followup",
      suggestedToolNames: selectedToolNames,
      originatingUserText: promptText,
    });
    const focusedFileIds = importedSelection.uploadedFiles.map((file) => file.file_id);
    const partitionedUris = partitionBisqueRowsByUri(selection.selectedRows);
    return {
      promptForModel: promptText,
      selectedUploads: importedSelection.uploadedFiles,
      selectedFileIds: focusedFileIds,
      quickPreviewFileIds:
        selection.intent === "preview"
          ? importedSelection.uploadedFiles.map((file) => file.file_id)
          : [],
      resolvedRows: selection.selectedRows,
      selectedToolNames,
      selectionContext:
        focusedFileIds.length > 0 ||
        partitionedUris.resourceUris.length > 0 ||
        partitionedUris.datasetUris.length > 0
          ? {
              context_id: makeId(),
              source: "deictic_followup",
              focused_file_ids: focusedFileIds,
              resource_uris: partitionedUris.resourceUris,
              dataset_uris: partitionedUris.datasetUris,
              originating_message_id: null,
              originating_user_text: promptText,
              suggested_domain: conversation.activeSelectionContext?.suggested_domain ?? null,
              suggested_tool_names: selectedToolNames,
            }
          : null,
    };
  };

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = (): void => {
      const shouldUseDark =
        themePreference === "dark" ||
        (themePreference === "system" && mediaQuery.matches);
      document.documentElement.classList.toggle("dark", shouldUseDark);
      document.body.classList.toggle("dark", shouldUseDark);
      document.documentElement.style.colorScheme = shouldUseDark ? "dark" : "light";
      document.body.style.colorScheme = shouldUseDark ? "dark" : "light";
      document.documentElement.setAttribute(
        "data-theme",
        shouldUseDark ? "dark" : "light"
      );
      document.body.setAttribute("data-theme", shouldUseDark ? "dark" : "light");
      setResolvedTheme(shouldUseDark ? "dark" : "light");
    };
    applyTheme();
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", applyTheme);
    } else {
      mediaQuery.addListener(applyTheme);
    }
    return () => {
      if (typeof mediaQuery.removeEventListener === "function") {
        mediaQuery.removeEventListener("change", applyTheme);
      } else {
        mediaQuery.removeListener(applyTheme);
      }
    };
  }, [themePreference]);

  const activeMessages = activeConversation?.messages ?? EMPTY_UI_MESSAGES;
  const activeConversationHydrated = activeConversation?.hydrated ?? true;
  const activePrompt =
    activeConversation &&
    Object.prototype.hasOwnProperty.call(
      composerDraftsByConversationId,
      activeConversation.id
    )
      ? composerDraftsByConversationId[activeConversation.id] ?? ""
      : activeConversation?.prompt ?? "";
  const activePendingFiles = activeConversation?.pendingFiles ?? EMPTY_FILES;
  const activeAvailableUploadedFiles =
    activeConversation?.uploadedFiles ?? EMPTY_UPLOADED_FILES;
  const activeStagedUploadFileIds =
    activeConversation?.stagedUploadFileIds ?? EMPTY_STRING_ARRAY;
  const activeSelectionContext = activeConversation?.activeSelectionContext ?? null;
  const activeSelectionContextFileIds =
    activeSelectionContext?.focused_file_ids ?? EMPTY_STRING_ARRAY;
  const activeComposerWorkflowPreset = activeConversation?.composerWorkflowPreset ?? null;
  const isProModeComposerActive = activeComposerWorkflowPreset?.id === "pro_mode";
  const activeComposerIntelligenceMode: ComposerIntelligenceMode =
    isProModeComposerActive ? "pro" : "high";
  const activeUploadedFiles = useMemo(() => {
    const combinedFileIds = uniqueFileIds([
      ...activeStagedUploadFileIds,
      ...activeSelectionContextFileIds,
    ]);
    if (combinedFileIds.length === 0 || activeAvailableUploadedFiles.length === 0) {
      return [];
    }
    const byId = new Map(
      activeAvailableUploadedFiles.map((file) => [file.file_id, file] as const)
    );
    return combinedFileIds
      .map((fileId) => byId.get(fileId))
      .filter((file): file is UploadedFileRecord => Boolean(file));
  }, [activeAvailableUploadedFiles, activeSelectionContextFileIds, activeStagedUploadFileIds]);
  const activeFailedUploadPreviewIds =
    activeConversation?.failedUploadPreviewIds ?? EMPTY_FAILED_UPLOAD_PREVIEW_IDS;
  const activeBisqueLinksByFileId =
    activeConversation?.bisqueLinksByFileId ?? EMPTY_BISQUE_LINKS_BY_FILE_ID;
  const activeSending = Boolean(
    activeConversation?.sending || activeConversation?.streamingMessageId
  );
  const activeChatError = activeConversation?.chatError ?? null;
  const activeStreamingMessageId = activeConversation?.streamingMessageId ?? null;
  const activeStreamingMessage = useMemo(
    () =>
      activeStreamingMessageId
        ? activeMessages.find((message) => message.id === activeStreamingMessageId) ?? null
        : null,
    [activeMessages, activeStreamingMessageId]
  );
  const activeStreamingRunId = activeStreamingMessage?.runId ?? null;
  // Live token total for the in-flight turn, summed from run.token_usage events
  // as each model call completes. Reuses the same dedupe-and-sum logic as the
  // completed-message footer so the ticker and the final count agree.
  const activeStreamingTokenUsage = useMemo(
    () =>
      activeStreamingMessage
        ? extractRunTokenUsage({
            responseMetadata: activeStreamingMessage.responseMetadata,
            runEvents: activeStreamingMessage.runEvents,
          })
        : null,
    [activeStreamingMessage]
  );
  const shouldShowBlankChatUsage =
    authStatus === "authenticated" &&
    activePanel === "chat" &&
    activeConversationHydrated &&
    activeMessages.length === 0;
  const blankChatUsageKey = `${authMode ?? ""}:${authUser ?? ""}`;
  useEffect(() => {
    blankChatUsageRequestRef.current = {
      key: blankChatUsageKey,
      inFlight: false,
      loaded: false,
      failed: false,
    };
    return queueEffectUpdate(() => {
      setBlankChatTokenUsage(null);
      setBlankChatUsageError(null);
      setBlankChatUsageLoading(false);
    });
  }, [blankChatUsageKey]);
  useEffect(() => {
    const requestState = blankChatUsageRequestRef.current;
    if (!shouldShowBlankChatUsage || requestState.key !== blankChatUsageKey) {
      return;
    }
    if (requestState.inFlight || requestState.loaded || requestState.failed) {
      return;
    }
    let cancelled = false;
    requestState.inFlight = true;
    const cancelQueuedLoading = queueEffectUpdate(() => {
      if (cancelled) {
        return;
      }
      setBlankChatUsageLoading(true);
      setBlankChatUsageError(null);
    });
    void loadCurrentUserTokenUsage(365)
      .then((response) => {
        if (!cancelled) {
          requestState.loaded = true;
          setBlankChatTokenUsage(response);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          requestState.failed = true;
          setBlankChatUsageError(normalizeApiError(error));
        }
      })
      .finally(() => {
        requestState.inFlight = false;
        if (!cancelled) {
          setBlankChatUsageLoading(false);
        }
      });
    return () => {
      cancelled = true;
      requestState.inFlight = false;
      cancelQueuedLoading();
    };
  }, [blankChatUsageKey, loadCurrentUserTokenUsage, shouldShowBlankChatUsage]);
  const requestChatScrollToBottom = useCallback((): void => {
    setChatScrollRequestKey((current) => current + 1);
  }, []);
  const activeConversationStopId = activeConversation?.id ?? null;
  // Remember the most recent backend run id seen for each conversation so the
  // Stop button can cancel the server-side run even if the streaming message's
  // run id is momentarily unavailable (e.g. just after a refresh re-attach).
  const activeRunIdByConversationRef = useRef<Map<string, string>>(new Map());
  useEffect(() => {
    if (!activeConversationStopId || !activeStreamingRunId) {
      return;
    }
    activeRunIdByConversationRef.current.set(activeConversationStopId, activeStreamingRunId);
  }, [activeConversationStopId, activeStreamingRunId]);
  const stopActiveConversation = useCallback((): void => {
    if (!activeConversationStopId) {
      return;
    }
    const runIdToCancel =
      activeStreamingRunId ||
      activeRunIdByConversationRef.current.get(activeConversationStopId) ||
      "";
    requestStopConversation(activeConversationStopId);
    if (runIdToCancel) {
      void apiClient.cancelRun(runIdToCancel, "Stopped from chat composer").catch(() => {
        // The run may already be terminal; the local stream is already stopped.
      });
    }
  }, [activeConversationStopId, activeStreamingRunId, apiClient, requestStopConversation]);

  useEffect(() => {
    const conversationId = activeConversation?.id ?? null;
    const messageId = activeStreamingMessageId;
    const runId = activeStreamingRunId;
    if (!conversationId || !messageId || !runId) {
      return;
    }
    let cancelled = false;
    // Poll incrementally: keep a sequence cursor and accumulate events so each
    // tick only transfers events the panel does not already hold. Long runs
    // produce thousands of events; re-paging from zero every tick repeated
    // dozens of requests per second for no new information.
    let collectedEvents: RunEvent[] = [];
    let afterSequence = 0;

    const pollRunEvents = async (): Promise<void> => {
      try {
        const response = await listRunEvents(apiClient, runId, 200, { afterSequence });
        if (cancelled || response.events.length === 0) {
          return;
        }
        collectedEvents = [...collectedEvents, ...response.events];
        afterSequence = response.events.reduce((current, event) => {
          const sequence = Math.floor(Number(event.payload?.sequence) || 0);
          return sequence > current ? sequence : current;
        }, afterSequence);
        const snapshot = collectedEvents;
        updateConversation(conversationId, (current) => ({
          ...current,
          messages: current.messages.map((item) =>
            item.id === messageId
              ? {
                  ...item,
                  runEvents: snapshot,
                }
              : item
          ),
        }));
      } catch {
        // Non-blocking while the run is still streaming.
      }
    };

    void pollRunEvents();
    const intervalId = window.setInterval(() => {
      void pollRunEvents();
    }, 1250);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
	  }, [
	    activeConversation?.id,
	    activeStreamingMessageId,
	    activeStreamingRunId,
	    apiClient,
	    updateConversation,
	  ]);

  const viewerUploadedFiles =
    resourceViewerContext?.uploadedFiles ?? activeAvailableUploadedFiles;
  const viewerBisqueLinksByFileId =
    resourceViewerContext?.bisqueLinksByFileId ?? activeBisqueLinksByFileId;

  const pendingPreviewFiles = useMemo(
    () =>
      activePendingFiles.map((file, index) => {
        const canPreviewInBrowser = supportsBrowserPreview(file.name, file.type);
        return {
          key: `${file.name}-${file.size}-${file.lastModified}-${index}`,
          name: file.name,
          sizeLabel: formatBytes(file.size),
          canPreviewInBrowser,
          isScientific: isScientificUpload(file.name),
          objectUrl: canPreviewInBrowser ? URL.createObjectURL(file) : null,
        };
      }),
    [activePendingFiles]
  );

  useEffect(() => {
    return () => {
      pendingPreviewFiles.forEach((file) => {
        if (file.objectUrl) {
          URL.revokeObjectURL(file.objectUrl);
        }
      });
    };
  }, [pendingPreviewFiles]);

  const uploadedPreviewFiles = useMemo(
    () =>
      activeUploadedFiles.map((file) => {
        const hasFailedPreview = Boolean(activeFailedUploadPreviewIds[file.file_id]);
        const canPreview =
          !hasFailedPreview &&
          (isScientificUpload(file.original_name) ||
            supportsBrowserPreview(file.original_name, file.content_type));
        return {
          id: file.file_id,
          name: file.original_name,
          sizeLabel: formatBytes(file.size_bytes),
          isScientific: isScientificUpload(file.original_name),
          previewUrl: canPreview ? apiClient.uploadPreviewUrl(file.file_id) : null,
        };
      }),
    [activeFailedUploadPreviewIds, activeUploadedFiles, apiClient]
  );
  const hasComposerAttachedFiles =
    activePendingFiles.length > 0 ||
    activeStagedUploadFileIds.length > 0 ||
    activeSelectionContextFileIds.length > 0;
  const selectedComposerResourceIds = useMemo(
    () => new Set(Object.keys(composerResourcePickerSelection)),
    [composerResourcePickerSelection]
  );
  const slashWorkflowQuery = normalizeSlashWorkflowQuery(activePrompt);
  const filteredSlashWorkflows = useMemo(
    () => composerWorkflows?.filterComposerWorkflows(slashWorkflowQuery) ?? [],
    [composerWorkflows, slashWorkflowQuery]
  );
  const slashMenuOpen =
    !composerResourcePickerOpen &&
    // Allow slash selection even when a persistent workflow such as Pro Mode
    // is already active so the menu still works as a workflow switcher.
    activePrompt.startsWith("/") &&
    activePrompt !== dismissedSlashPrompt;

  useEffect(() => {
    if (!slashMenuOpen || composerWorkflows) {
      return;
    }
    let cancelled = false;
    void loadComposerWorkflows()
      .then((module) => {
        if (!cancelled) {
          setComposerWorkflows(module);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [composerWorkflows, slashMenuOpen]);

  const resolvedActiveSlashWorkflowId = useMemo(() => {
    if (!slashMenuOpen || filteredSlashWorkflows.length === 0) {
      return null;
    }
    if (activeSlashWorkflowId && filteredSlashWorkflows.some((workflow) => workflow.id === activeSlashWorkflowId)) {
      return activeSlashWorkflowId;
    }
    return filteredSlashWorkflows[0]?.id ?? null;
  }, [activeSlashWorkflowId, filteredSlashWorkflows, slashMenuOpen]);

  const resolvedActiveComposerResourceId = useMemo(() => {
    if (!composerResourcePickerOpen || composerResources.length === 0) {
      return null;
    }
    if (activeComposerResourceId && composerResources.some((resource) => resource.file_id === activeComposerResourceId)) {
      return activeComposerResourceId;
    }
    return composerResources[0]?.file_id ?? null;
  }, [activeComposerResourceId, composerResourcePickerOpen, composerResources]);

  const refreshResources = useCallback((): void => {
    setResourceRefreshToken((value) => value + 1);
    setResourceCollectionRefreshToken((value) => value + 1);
  }, []);

  const openResourceCollection = useCallback((collection: ResourceCollectionRecord): void => {
    const collectionId = String(collection.collection_id || "").trim();
    if (!collectionId) {
      return;
    }
    setResourcesError(null);
    setActiveResourceCollectionId(collectionId);
    setActiveResourceCollectionSnapshot(collection);
  }, []);

  const clearActiveResourceCollection = useCallback((): void => {
    setActiveResourceCollectionId(null);
    setActiveResourceCollectionSnapshot(null);
  }, []);

  const updateResourceQuery = useCallback((value: string): void => {
    setResourceQuery(value);
  }, []);

  const updateResourceKindFilter = useCallback((value: ResourceKindFilter): void => {
    setResourceKindFilter(value);
  }, []);

  const updateResourceSourceFilter = useCallback((value: ResourceSourceFilter): void => {
    setResourceSourceFilter(value);
  }, []);

  const updateResourceSharingFilter = useCallback((value: ResourceSharingFilter): void => {
    setResourceSharingFilter(value);
  }, []);

  const updateResourceStatusFilter = useCallback((value: ResourceStatusFilter): void => {
    setResourceStatusFilter(value);
    if (value === "deleted") {
      setActiveResourceCollectionId(null);
      setActiveResourceCollectionSnapshot(null);
    }
  }, []);

  const updateResourceTagFilter = useCallback((value: string): void => {
    setResourceTagFilter(value);
  }, []);

  const flushResourceUploadProgress = useCallback((): void => {
    resourceUploadProgressBatcherRef.current?.flush();
  }, []);

  const updateResourceUploadProgress = useCallback((event: UploadProgressEvent): void => {
    void persistResourceUploadProgressEvent(resourceUploadQueueStore, event);
    resourceUploadProgressBatcherRef.current?.enqueue(event);
  }, []);

  const uploadResourceFiles = useCallback(
    async (files: File[], context?: ResourceUploadReselectionContext): Promise<void> => {
      const selectedFiles = files.filter((file) => file.size >= 0);
      if (selectedFiles.length === 0 || resourcesUploading) {
        return;
      }
      const resumeFrom = context?.resumeFrom;
      const resumeSession =
        selectedFiles.length === 1 && resumeFrom?.sessionId && resumeFrom.fileToken
          ? {
              sessionId: resumeFrom.sessionId,
              fileToken: resumeFrom.fileToken,
              progressId: resumeFrom.id,
            }
          : undefined;
      const uploadTargetCollection = context?.uploadTargetCollection ?? activeResourceCollection;
      const activeUploadCollectionId = String(uploadTargetCollection?.collection_id ?? "").trim();
      pausedResourceUploadSessionIdsRef.current.clear();
      setResourcesUploading(true);
      setResourcesError(null);
      try {
        const response = await apiClient.uploadFiles(selectedFiles, {
          onProgress: updateResourceUploadProgress,
          resumeSession,
          pauseSignal: {
            isPaused: (sessionId: string) => pausedResourceUploadSessionIdsRef.current.has(sessionId),
          },
        });
        const uploadedFileIds = uniqueFileIds(response.uploaded.map((file) => file.file_id));
        if (activeUploadCollectionId && uploadedFileIds.length > 0) {
          try {
            const collectionResponse = await apiClient.addResourcesToCollection(
              activeUploadCollectionId,
              uploadedFileIds,
              {
                source: "resources_folder_upload",
              }
            );
            setResourceCollections((previous) =>
              previous.map((collection) =>
                collection.collection_id === activeUploadCollectionId
                  ? collectionResponse.collection
                  : collection
              )
            );
            setActiveResourceCollectionSnapshot((current) =>
              current?.collection_id === activeUploadCollectionId
                ? collectionResponse.collection
                : current
            );
          } catch (error) {
            setResourcesError(`Uploaded, but could not add to folder: ${normalizeApiError(error)}`);
          }
        }
        flushResourceUploadProgress();
        setResourceUploadProgress((current) => {
          const next = current.filter((item) => item.status !== "completed");
          writeResourceUploadProgressToStorage(next);
          return next;
        });
        setResourceRefreshToken((value) => value + 1);
        setResourceCollectionRefreshToken((value) => value + 1);
      } catch (error) {
        if (error instanceof UploadPausedError) {
          return;
        }
        setResourcesError(normalizeApiError(error));
      } finally {
        flushResourceUploadProgress();
        setResourcesUploading(false);
      }
    },
    [
      activeResourceCollection,
      apiClient,
      flushResourceUploadProgress,
      resourcesUploading,
      updateResourceUploadProgress,
    ]
  );

  const dismissResourceUploadProgress = useCallback((item: ResourceUploadProgress): void => {
    void resourceUploadQueueStore.remove(item.id);
    setResourceUploadProgress((current) => {
      const next = current.filter((progressItem) => progressItem.id !== item.id);
      writeResourceUploadProgressToStorage(next);
      return next;
    });
  }, []);

  const pauseResourceUploadProgress = useCallback(
    async (item: ResourceUploadProgress): Promise<void> => {
      const sessionId = String(item.sessionId ?? "").trim();
      if (!sessionId) {
        return;
      }
      pausedResourceUploadSessionIdsRef.current.add(sessionId);
      updateResourceUploadProgress({
        id: item.id,
        fileName: item.fileName,
        fileIndex: 0,
        fileToken: item.fileToken ?? undefined,
        sessionId,
        fingerprint: item.fingerprint ?? undefined,
        relativePath: item.relativePath ?? undefined,
        status: "paused",
        totalBytes: item.totalBytes,
        bytesVerified: item.bytesVerified,
        bytesCommitted: 0,
      });
      setResourcesError(null);
      try {
        const response = await apiClient.pauseUploadSession(sessionId);
        updateResourceUploadProgress({
          id: item.id,
          fileName: item.fileName,
          fileIndex: 0,
          fileToken: item.fileToken ?? undefined,
          sessionId,
          fingerprint: item.fingerprint ?? undefined,
          relativePath: item.relativePath ?? undefined,
          status: "paused",
          totalBytes: item.totalBytes,
          bytesVerified: Math.max(item.bytesVerified, response.session.bytes_verified ?? 0),
          bytesCommitted: response.session.bytes_committed ?? 0,
        });
      } catch (error) {
        pausedResourceUploadSessionIdsRef.current.delete(sessionId);
        setResourcesError(normalizeApiError(error));
      }
    },
    [apiClient, updateResourceUploadProgress]
  );

  const cancelResourceUploadProgress = useCallback(
    async (item: ResourceUploadProgress): Promise<void> => {
      const sessionId = String(item.sessionId ?? "").trim();
      if (!sessionId) {
        dismissResourceUploadProgress(item);
        return;
      }
      pausedResourceUploadSessionIdsRef.current.delete(sessionId);
      setResourcesError(null);
      try {
        await apiClient.cancelUploadSession(sessionId);
        dismissResourceUploadProgress(item);
      } catch (error) {
        setResourcesError(normalizeApiError(error));
      }
    },
    [apiClient, dismissResourceUploadProgress]
  );

  const createResourceFolderFromSelection = useCallback(
    async (request: ResourceCollectionSelectionRequest): Promise<void> => {
      const folderName = request.name.trim();
      const resourceIds = Array.from(
        new Set(request.resourceIds.map((fileId) => fileId.trim()).filter(Boolean))
      );
      if (!folderName) {
        return;
      }
      const hasResources = resourceIds.length > 0;
      const source = hasResources ? "resources_bulk_toolbar" : "resources_new_folder";
      setResourcesError(null);
      try {
        const collectionResponse = await apiClient.createResourceCollection({
          name: folderName,
          collection_type: "folder",
          metadata: {
            source,
            selected_resource_count: resourceIds.length,
          },
        });
        if (hasResources) {
          await apiClient.addResourcesToCollection(
            collectionResponse.collection.collection_id,
            resourceIds,
            {
              source,
            }
          );
        }
        setResourceRefreshToken((value) => value + 1);
        setResourceCollectionRefreshToken((value) => value + 1);
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [apiClient]
  );

  const addResourcesToFolderFromSelection = useCallback(
    async (request: ResourceCollectionAddSelectionRequest): Promise<void> => {
      const collectionId = request.collectionId.trim();
      const resourceIds = Array.from(
        new Set(request.resourceIds.map((fileId) => fileId.trim()).filter(Boolean))
      );
      if (!collectionId || resourceIds.length === 0) {
        return;
      }
      setResourcesError(null);
      try {
        await apiClient.addResourcesToCollection(collectionId, resourceIds, {
          source: "resources_bulk_toolbar",
        });
        setResourceRefreshToken((value) => value + 1);
        setResourceCollectionRefreshToken((value) => value + 1);
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [apiClient]
  );

  const renameResourceFromResources = useCallback(
    async (resource: ResourceRecord, name: string): Promise<void> => {
      const fileId = String(resource.file_id || "").trim();
      const nextName = name.trim();
      if (!fileId || !nextName || nextName === resource.original_name) {
        return;
      }
      setResourcesError(null);
      try {
        const response = await renameResourceRequest(apiClient, fileId, nextName);
        setResources((previous) =>
          previous.map((item) => (item.file_id === fileId ? response.resource : item))
        );
        setComposerResources((previous) =>
          previous.map((item) => (item.file_id === fileId ? response.resource : item))
        );
        setResourceRefreshToken((value) => value + 1);
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [apiClient]
  );

  const renameResourceCollectionFromResources = useCallback(
    async (collection: ResourceCollectionRecord, name: string): Promise<void> => {
      const collectionId = String(collection.collection_id || "").trim();
      const nextName = name.trim();
      if (!collectionId || !nextName || nextName === collection.name) {
        return;
      }
      setResourcesError(null);
      try {
        const response = await renameResourceCollectionRequest(apiClient, collectionId, nextName);
        setResourceCollections((previous) =>
          previous.map((item) =>
            item.collection_id === collectionId ? response.collection : item
          )
        );
        setActiveResourceCollectionSnapshot((current) =>
          current?.collection_id === collectionId ? response.collection : current
        );
        setResourceCollectionRefreshToken((value) => value + 1);
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [apiClient]
  );

  const deleteResourceCollectionFromResources = useCallback(
    async (collection: ResourceCollectionRecord): Promise<void> => {
      const collectionId = String(collection.collection_id || "").trim();
      if (!collectionId) {
        return;
      }
      setResourcesError(null);
      try {
        await deleteResourceCollectionRequest(apiClient, collectionId);
        setResourceCollections((previous) =>
          previous.filter((item) => item.collection_id !== collectionId)
        );
        if (activeResourceCollectionId === collectionId) {
          setActiveResourceCollectionId(null);
          setActiveResourceCollectionSnapshot(null);
        }
        setResourceRefreshToken((value) => value + 1);
        setResourceCollectionRefreshToken((value) => value + 1);
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [activeResourceCollectionId, apiClient]
  );

  const removeResourceFromActiveCollection = useCallback(
    async (resource: ResourceRecord): Promise<void> => {
      const collectionId = String(activeResourceCollection?.collection_id ?? "").trim();
      const fileId = String(resource.file_id || "").trim();
      if (!collectionId || !fileId) {
        return;
      }
      setResourcesError(null);
      try {
        const response = await removeResourceFromCollectionRequest(apiClient, collectionId, fileId);
        setResources((previous) => previous.filter((item) => item.file_id !== fileId));
        setResourceTotalCount((value) => Math.max(0, value - 1));
        setResourceCollections((previous) =>
          previous.map((item) =>
            item.collection_id === collectionId ? response.collection : item
          )
        );
        setActiveResourceCollectionSnapshot((current) =>
          current?.collection_id === collectionId ? response.collection : current
        );
        setResourceCollectionRefreshToken((value) => value + 1);
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [activeResourceCollection?.collection_id, apiClient]
  );

  const pushResourceToBisque = useCallback(
    async (resource: ResourceRecord): Promise<void> => {
      const fileId = String(resource.file_id || "").trim();
      if (!fileId) {
        return;
      }
      if (!bisqueCredentialsLinked) {
        setResourcesError(
          "Link your BisQue account in Settings before pushing resources to BisQue."
        );
        return;
      }
      setResourcesError(null);
      try {
        const response = await apiClient.pushResourcesToBisque({ fileIds: [fileId] });
        const uploaded = response.uploads[0];
        const viewUrl = String(uploaded?.client_view_url ?? "").trim();
        showSuccessToast(`Pushed "${resource.original_name}" to BisQue`, {
          description: uploaded?.resource_uri ?? undefined,
          action: viewUrl
            ? {
                label: "View in BisQue",
                onClick: () => {
                  window.open(viewUrl, "_blank", "noopener,noreferrer");
                },
              }
            : undefined,
        });
      } catch (error) {
        setResourcesError(normalizeApiError(error));
      }
    },
    [apiClient, bisqueCredentialsLinked]
  );

  const pushCollectionToBisque = useCallback(
    async (collection: ResourceCollectionRecord): Promise<void> => {
      const collectionId = String(collection.collection_id || "").trim();
      if (!collectionId) {
        return;
      }
      if (!bisqueCredentialsLinked) {
        setResourcesError(
          "Link your BisQue account in Settings before pushing folders to BisQue."
        );
        return;
      }
      setResourcesError(null);
      try {
        const response = await apiClient.pushResourcesToBisque({
          collectionIds: [collectionId],
        });
        const dataset = response.datasets[0];
        const viewUrl = String(dataset?.client_view_url ?? "").trim();
        const fileCount = response.uploads.length;
        showSuccessToast(
          `Pushed folder "${collection.name}" to BisQue as a dataset`,
          {
            description: `${fileCount} ${fileCount === 1 ? "file" : "files"} uploaded · dataset "${
              dataset?.name ?? collection.name
            }"`,
            action: viewUrl
              ? {
                  label: "View in BisQue",
                  onClick: () => {
                    window.open(viewUrl, "_blank", "noopener,noreferrer");
                  },
                }
              : undefined,
          }
        );
      } catch (error) {
        setResourcesError(normalizeApiError(error));
      }
    },
    [apiClient, bisqueCredentialsLinked]
  );

  const loadResourceShareGrantsFromResources = useCallback(
    async (resource: ResourceRecord): Promise<ResourceShareGrantRecord[]> => {
      const resourceId = String(resource.file_id ?? "").trim();
      if (!resourceId) {
        return [];
      }
      const response = await loadResourceShareGrantsRequest(apiClient, resourceId);
      return response.grants;
    },
    [apiClient]
  );

  const createResourceShareGrantFromResources = useCallback(
    async (
      resource: ResourceRecord,
      request: ResourceShareGrantRequest
    ): Promise<ResourceShareGrantRecord> => {
      const resourceId = String(resource.file_id ?? "").trim();
      if (!resourceId) {
        throw new Error("Resource id is required to grant access.");
      }
      setResourcesError(null);
      try {
        const response = await createResourceShareGrantRequest(apiClient, resourceId, request);
        return response.grant;
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [apiClient]
  );

  const createBulkResourceShareGrantsFromResources = useCallback(
    async (
      resources: ResourceRecord[],
      request: ResourceShareGrantRequest
    ): Promise<ResourceShareGrantRecord[]> => {
      const resourceIds = Array.from(
        new Set(resources.map((resource) => String(resource.file_id ?? "").trim()).filter(Boolean))
      );
      if (resourceIds.length === 0) {
        throw new Error("At least one resource id is required to grant access.");
      }
      setResourcesError(null);
      try {
        const response = await createBulkResourceShareGrantsRequest(apiClient, resourceIds, request);
        return response.grants;
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [apiClient]
  );

  const createResourceCollectionShareGrantsFromResources = useCallback(
    async (
      collection: ResourceCollectionRecord,
      request: ResourceShareGrantRequest
    ): Promise<ResourceShareGrantRecord[]> => {
      const collectionId = String(collection.collection_id ?? "").trim();
      if (!collectionId) {
        throw new Error("Collection id is required to grant folder access.");
      }
      setResourcesError(null);
      try {
        const response = await createResourceCollectionShareGrantsRequest(
          apiClient,
          collectionId,
          request
        );
        return response.grants;
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [apiClient]
  );

  const revokeResourceShareGrantFromResources = useCallback(
    async (
      resource: ResourceRecord,
      grant: ResourceShareGrantRecord
    ): Promise<ResourceShareGrantRecord> => {
      const resourceId = String(resource.file_id ?? "").trim();
      const grantId = String(grant.grant_id ?? "").trim();
      if (!resourceId || !grantId) {
        throw new Error("Resource id and grant id are required to revoke access.");
      }
      setResourcesError(null);
      try {
        const response = await revokeResourceShareGrantRequest(apiClient, resourceId, grantId);
        return response.grant;
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [apiClient]
  );

  const resourceHasMore = resources.length < resourceTotalCount;

  const loadMoreResources = useCallback((): void => {
    if (
      authStatus !== "authenticated" ||
      resourcesLoading ||
      resourcesLoadingMore ||
      resources.length >= resourceTotalCount
    ) {
      return;
    }
    const offset = resources.length;
    const activeQuery = debouncedResourceQuery.trim();
    const activeTagFilters = parseResourceTagFilter(resourceTagFilter);
    const activeCollectionId = activeResourceCollection?.collection_id ?? "";
    const activeResourceListKey = [
      activeCollectionId ? `folder:${activeCollectionId}` : "library",
      activeQuery,
      resourceKindFilter,
      resourceSourceFilter,
      resourceSharingFilter,
      resourceStatusFilter,
      activeTagFilters.join("\u0001"),
      String(resourceRefreshToken),
    ].join("\u0000");
    setResourcesLoadingMore(true);
    const request = activeCollectionId
      ? loadResourceFolderResources(apiClient, activeCollectionId, {
          limit: RESOURCE_PAGE_SIZE,
          offset,
          query: activeQuery || undefined,
          kind: resourceKindFilter,
          source: resourceSourceFilter,
          sharing: resourceSharingFilter,
          status: resourceStatusFilter,
          tags: activeTagFilters,
        })
      : loadLibraryResources(apiClient, {
          limit: RESOURCE_PAGE_SIZE,
          offset,
          query: activeQuery || undefined,
          kind: resourceKindFilter,
          source: resourceSourceFilter,
          sharing: resourceSharingFilter,
          status: resourceStatusFilter,
          tags: activeTagFilters,
        });
    void request
      .then((payload) => {
        if (resourceListKeyRef.current !== activeResourceListKey) {
          return;
        }
        setResources((previous) => {
          const seen = new Set(previous.map((resource) => resource.file_id));
          const merged = [...previous];
          payload.resources.forEach((resource) => {
            if (!seen.has(resource.file_id)) {
              seen.add(resource.file_id);
              merged.push(resource);
            }
          });
          return merged;
        });
        setResourceTotalCount(Math.max(0, Math.floor(Number(payload.count) || 0)));
      })
      .catch((error) => {
        if (resourceListKeyRef.current !== activeResourceListKey) {
          return;
        }
        setResourcesError(normalizeApiError(error));
      })
      .finally(() => {
        if (resourceListKeyRef.current !== activeResourceListKey) {
          return;
        }
        setResourcesLoadingMore(false);
      });
  }, [
    apiClient,
    activeResourceCollection?.collection_id,
    authStatus,
    debouncedResourceQuery,
    resourceKindFilter,
    resourceRefreshToken,
    resourceSharingFilter,
    resourceSourceFilter,
    resourceStatusFilter,
    resourceTagFilter,
    resourceTotalCount,
    resources.length,
    resourcesLoading,
    resourcesLoadingMore,
  ]);

  const refreshAdminConsole = (): void => {
    setAdminRefreshToken((value) => value + 1);
  };

  const createAdminConsoleOrganization = async (
    payload: AdminCreateOrganizationRequest
  ): Promise<void> => {
    try {
      await createAdminOrganization(apiClient, payload);
      setAdminError(null);
      refreshAdminConsole();
    } catch (error) {
      const message = normalizeApiError(error);
      setAdminError(message);
      throw Object.assign(new Error(message), { cause: error });
    }
  };

  const createAdminConsoleUser = async (payload: AdminCreateUserRequest): Promise<void> => {
    try {
      await createAdminUser(apiClient, payload);
      setAdminError(null);
      refreshAdminConsole();
    } catch (error) {
      const message = normalizeApiError(error);
      setAdminError(message);
      throw Object.assign(new Error(message), { cause: error });
    }
  };

  const deleteAdminConsoleUser = async (userId: string): Promise<void> => {
    const key = String(userId || "").trim();
    if (!key) {
      return;
    }
    setAdminUserDeletingById((previous) => ({ ...previous, [key]: true }));
    try {
      await deleteAdminUser(apiClient, key);
      setAdminError(null);
      refreshAdminConsole();
    } catch (error) {
      setAdminError(normalizeApiError(error));
    } finally {
      setAdminUserDeletingById((previous) => {
        const next = { ...previous };
        delete next[key];
        return next;
      });
    }
  };

  const updateAdminConsoleUserStatus = async (
    userId: string,
    status: AdminUserStatus
  ): Promise<void> => {
    const key = String(userId || "").trim();
    if (!key) {
      return;
    }
    setAdminUserUpdatingById((previous) => ({ ...previous, [key]: true }));
    try {
      await updateAdminUserStatus(apiClient, key, status);
      setAdminError(null);
      refreshAdminConsole();
    } catch (error) {
      setAdminError(normalizeApiError(error));
    } finally {
      setAdminUserUpdatingById((previous) => {
        const next = { ...previous };
        delete next[key];
        return next;
      });
    }
  };

  const inspectAdminRunEvents = async (runId: string): Promise<void> => {
    const key = String(runId || "").trim();
    if (!key) {
      return;
    }
    setActiveAdminRunEventRunId(key);
    setAdminRunEventsLoadingById((previous) => ({ ...previous, [key]: true }));
    try {
      const payload = await listRunEvents(apiClient, key, 200);
      setAdminRunEventsById((previous) => ({ ...previous, [key]: payload.events }));
      setAdminError(null);
    } catch (error) {
      setAdminError(normalizeApiError(error));
    } finally {
      setAdminRunEventsLoadingById((previous) => {
        const next = { ...previous };
        delete next[key];
        return next;
      });
    }
  };

  const openAdminPanel = (): void => {
    if (!authIsAdmin) {
      return;
    }
    rememberActiveConversationScrollPosition();
    setActivePanel("admin");
    setViewerOpen(false);
    setResourceViewerContext(null);
    refreshAdminConsole();
  };

  const cancelAdminRun = async (runId: string): Promise<void> => {
    const key = String(runId || "").trim();
    if (!key) {
      return;
    }
    setAdminRunCancellingById((previous) => ({ ...previous, [key]: true }));
    try {
      await apiClient.cancelAdminRun(key);
      setAdminError(null);
      refreshAdminConsole();
    } catch (error) {
      setAdminError(normalizeApiError(error));
    } finally {
      setAdminRunCancellingById((previous) => {
        const next = { ...previous };
        delete next[key];
        return next;
      });
    }
  };

  const requeueAdminRun = async (runId: string): Promise<void> => {
    const key = String(runId || "").trim();
    if (!key) {
      return;
    }
    setAdminRunRequeueingById((previous) => ({ ...previous, [key]: true }));
    try {
      await apiClient.requeueAdminRun(key, "admin requeue");
      setAdminError(null);
      refreshAdminConsole();
    } catch (error) {
      setAdminError(normalizeApiError(error));
    } finally {
      setAdminRunRequeueingById((previous) => {
        const next = { ...previous };
        delete next[key];
        return next;
      });
    }
  };

  const deleteAdminConversation = async (
    conversationId: string,
    userId: string
  ): Promise<void> => {
    const conversationKey = `${userId}:${conversationId}`;
    setAdminDeletingConversationKey(conversationKey);
    try {
      await apiClient.deleteAdminConversation(conversationId, userId);
      setAdminError(null);
      if (authUser && userId === `bisque:${authUser.toLowerCase()}`) {
        setConversations((previous) =>
          previous.filter((conversation) => conversation.id !== conversationId)
        );
        if (activeConversationId === conversationId) {
          setActiveConversationId(null);
        }
      }
      refreshAdminConsole();
    } catch (error) {
      setAdminError(normalizeApiError(error));
    } finally {
      setAdminDeletingConversationKey(null);
    }
  };

  const resourceToUploadedFile = (resource: ResourceRecord): UploadedFileRecord => ({
    file_id: resource.file_id,
    original_name: resource.original_name,
    content_type: resource.content_type ?? null,
    size_bytes: Math.max(0, Number(resource.size_bytes) || 0),
    sha256: resource.sha256,
    created_at: resource.created_at,
  });

  const resourceToBisqueLink = (resource: ResourceRecord): BisqueViewerLink | null => {
    const clientViewUrl = String(resource.client_view_url ?? "").trim();
    if (!clientViewUrl) {
      return null;
    }
    return {
      clientViewUrl,
      resourceUri: resource.source_uri ?? null,
      imageServiceUrl: resource.image_service_url ?? null,
      inputUrl: resource.source_uri ?? undefined,
    };
  };

  const openUploadedFilesInViewer = useCallback((
    selectedFiles: UploadedFileRecord[],
    selectedLinksByFileId: Record<string, BisqueViewerLink>
  ): void => {
    if (selectedFiles.length === 0) {
      return;
    }
    setResourceViewerContext({
      uploadedFiles: uniqueByFileId(selectedFiles),
      bisqueLinksByFileId: selectedLinksByFileId,
    });
    rememberActiveConversationScrollPosition();
    setActivePanel("scientific-viewer");
    setViewerOpen(false);
    setResourceRefreshToken((value) => value + 1);
  }, [rememberActiveConversationScrollPosition]);

  const openConversationFilesInViewer = useCallback((fileIds: string[]): void => {
    const selectedFileIds = uniqueFileIds(fileIds);
    if (selectedFileIds.length === 0) {
      return;
    }
    const selectedUploads = selectedFileIds
      .map((fileId) =>
        activeAvailableUploadedFiles.find((file) => file.file_id === fileId) ?? null
      )
      .filter((file): file is UploadedFileRecord => file !== null);
    if (selectedUploads.length === 0) {
      return;
    }
    const selectedLinks = Object.fromEntries(
      selectedUploads
        .map((file) => {
          const link = activeBisqueLinksByFileId[file.file_id];
          return link ? ([file.file_id, link] as const) : null;
        })
        .filter((entry): entry is readonly [string, BisqueViewerLink] => entry !== null)
    );
    openUploadedFilesInViewer(selectedUploads, selectedLinks);
  }, [activeAvailableUploadedFiles, activeBisqueLinksByFileId, openUploadedFilesInViewer]);

  const openResourceInViewer = (resource: ResourceRecord): void => {
    const uploaded = resourceToUploadedFile(resource);
    const bisqueLink = resourceToBisqueLink(resource);
    openUploadedFilesInViewer([uploaded], bisqueLink ? { [uploaded.file_id]: bisqueLink } : {});
  };

  // --- URL-as-navigation-state -------------------------------------------------------
  // The app has no router; navigation is React state. Reflect the active panel + open
  // Lens resource in the URL so the browser Back/Forward buttons work, a refresh
  // restores the view, and a Lens view is a shareable deep link. This coexists with the
  // ?conversation= sync — buildNavUrl preserves every other param, so each layer only
  // ever touches its own keys.
  const viewerResourceFileIds = useMemo(
    () => (resourceViewerContext?.uploadedFiles ?? []).map((file) => file.file_id),
    [resourceViewerContext]
  );
  const initialNavRef = useRef<NavState>(
    typeof window === "undefined"
      ? { panel: "chat", resourceFileIds: [] }
      : parseNavFromSearch(window.location.search)
  );
  const navRestoredRef = useRef(false);
  const suppressNavSyncRef = useRef(false);
  const lastNavKeyRef = useRef<string | null>(null);

  // Rebuild the Lens viewer context from resource file id(s) (deep link / Back / refresh).
  // Always fetches fresh by id (cheap, only on navigation) so it doesn't depend on the
  // in-memory list and stays referentially stable. Mirrors resourceToUploadedFile /
  // resourceToBisqueLink.
  const restoreViewerContextForFileIds = useCallback(
    async (fileIds: string[]): Promise<void> => {
      const ids = uniqueFileIds(fileIds);
      if (ids.length === 0) {
        return;
      }
      const records = await Promise.all(ids.map((id) => apiClient.getResource(id).catch(() => null)));
      const found = records.filter((record): record is ResourceRecord => record !== null);
      if (found.length === 0) {
        return;
      }
      const uploadedFiles = uniqueByFileId(
        found.map((record) => ({
          file_id: record.file_id,
          original_name: record.original_name,
          content_type: record.content_type ?? null,
          size_bytes: Math.max(0, Number(record.size_bytes) || 0),
          sha256: record.sha256,
          created_at: record.created_at,
        }))
      );
      const bisqueLinksByFileId: Record<string, BisqueViewerLink> = {};
      for (const record of found) {
        const clientViewUrl = String(record.client_view_url ?? "").trim();
        if (clientViewUrl) {
          bisqueLinksByFileId[record.file_id] = {
            clientViewUrl,
            resourceUri: record.source_uri ?? null,
            imageServiceUrl: record.image_service_url ?? null,
            inputUrl: record.source_uri ?? undefined,
          };
        }
      }
      setResourceViewerContext({ uploadedFiles, bisqueLinksByFileId });
    },
    [apiClient]
  );

  // One-time restore on load: apply a deep-linked panel + Lens resource once authenticated.
  useEffect(() => {
    if (navRestoredRef.current || authStatus !== "authenticated") {
      return;
    }
    navRestoredRef.current = true;
    const initial = initialNavRef.current;
    if (initial.panel !== "chat") {
      setActivePanel(initial.panel);
    }
    if (initial.panel === "scientific-viewer" && initial.resourceFileIds.length > 0) {
      void restoreViewerContextForFileIds(initial.resourceFileIds);
    }
  }, [authStatus, restoreViewerContextForFileIds]);

  // State -> URL: push a history entry on each navigation (so Back reverses it), replace
  // on the first sync, and skip writes that originated from Back/Forward (popstate).
  useEffect(() => {
    if (typeof window === "undefined" || !navRestoredRef.current || authStatus !== "authenticated") {
      return;
    }
    const nav: NavState = { panel: activePanel, resourceFileIds: viewerResourceFileIds };
    const key = navStateKey(nav);
    if (key === lastNavKeyRef.current) {
      return;
    }
    const isFirstSync = lastNavKeyRef.current === null;
    lastNavKeyRef.current = key;
    if (suppressNavSyncRef.current) {
      suppressNavSyncRef.current = false;
      return;
    }
    const nextUrl = buildNavUrl(
      { pathname: window.location.pathname, search: window.location.search, hash: window.location.hash },
      nav
    );
    const currentRelativeUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextUrl === currentRelativeUrl) {
      return;
    }
    if (isFirstSync) {
      window.history.replaceState(window.history.state, "", nextUrl);
    } else {
      window.history.pushState({}, "", nextUrl);
    }
  }, [activePanel, viewerResourceFileIds, authStatus]);

  // Back/Forward: restore the panel + Lens resource from the URL the browser navigated to.
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const handlePopState = (): void => {
      const nav = parseNavFromSearch(window.location.search);
      suppressNavSyncRef.current = true;
      setActivePanel(nav.panel);
      if (nav.panel === "scientific-viewer" && nav.resourceFileIds.length > 0) {
        void restoreViewerContextForFileIds(nav.resourceFileIds);
      }
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [restoreViewerContextForFileIds]);

  const stageResourcesForConversation = (
    conversationId: string,
    resourcesToStage: ResourceRecord[],
    options?: {
      persistSelectionContext?: boolean;
      source?: string;
      suggestedDomain?: string | null;
      suggestedToolNames?: string[];
      originatingMessageId?: string | null;
    }
  ): void => {
    const stagedResources = resourcesToStage.filter(
      (resource, index, all) =>
        resource.file_id.trim().length > 0 &&
        all.findIndex((item) => item.file_id === resource.file_id) === index
    );
    if (!conversationId.trim() || stagedResources.length === 0) {
      return;
    }
    const uploadedFiles = stagedResources.map(resourceToUploadedFile);
    const bisqueLinks = Object.fromEntries(
      stagedResources
        .map((resource) => {
          const bisqueLink = resourceToBisqueLink(resource);
          return bisqueLink ? ([resource.file_id, bisqueLink] as const) : null;
        })
      .filter((entry): entry is readonly [string, BisqueViewerLink] => entry !== null)
    );
    updateConversation(conversationId, (conversation) => {
      const latestUserMessage = [...conversation.messages]
        .reverse()
        .find((message) => message.role === "user" && message.content.trim().length > 0);
      const stagedSelectionUris = stagedResources
        .map((resource) =>
          String(
            resource.canonical_resource_uri ??
              resource.source_uri ??
              resourceToBisqueLink(resource)?.resourceUri ??
              ""
          ).trim()
        )
        .filter((value) => value.length > 0);
      const partitionedStagedUris = partitionBisqueUris(
        stagedSelectionUris,
        stagedResources
          .filter((resource) => String(resource.resource_kind ?? "").trim().toLowerCase() === "dataset")
          .map((resource) =>
            String(
              resource.canonical_resource_uri ??
                resource.source_uri ??
                resourceToBisqueLink(resource)?.resourceUri ??
                ""
            ).trim()
          )
          .filter((value) => value.length > 0)
      );
      const nextSelectionContext =
        options?.persistSelectionContext
          ? ({
              context_id: makeId(),
              source: options?.source ?? "use_in_chat",
              focused_file_ids: uploadedFiles.map((file) => file.file_id),
              resource_uris: partitionedStagedUris.resourceUris,
              dataset_uris: partitionedStagedUris.datasetUris,
              originating_message_id: options?.originatingMessageId ?? null,
              originating_user_text: latestUserMessage?.content?.trim() || null,
              suggested_domain: options?.suggestedDomain ?? null,
              suggested_tool_names: Array.from(
                new Set((options?.suggestedToolNames ?? []).map((name) => String(name || "").trim()))
              ).filter((name) => name.length > 0),
                } satisfies SelectionContext)
          : conversation.activeSelectionContext;
      return {
        ...conversation,
        updatedAt: Date.now(),
        uploadedFiles: uniqueByFileId([...conversation.uploadedFiles, ...uploadedFiles]),
        stagedUploadFileIds: uniqueFileIds([
          ...conversation.stagedUploadFileIds,
          ...uploadedFiles.map((file) => file.file_id),
        ]),
        activeSelectionContext: nextSelectionContext,
        bisqueLinksByFileId:
          Object.keys(bisqueLinks).length > 0
            ? {
                ...conversation.bisqueLinksByFileId,
                ...bisqueLinks,
              }
            : conversation.bisqueLinksByFileId,
        selectionImportPending: false,
      };
    });
    setActivePanel("chat");
    setResourceViewerContext(null);
  };

  const addResourceToActiveConversation = (resource: ResourceRecord): void => {
    if (!activeConversation) {
      return;
    }
    stageResourcesForConversation(activeConversation.id, [resource], {
      persistSelectionContext: true,
      source: "resource_browser",
    });
  };

  const toggleComposerResourceSelection = (resource: ResourceRecord): void => {
    setActiveComposerResourceId(resource.file_id);
    setComposerResourcePickerSelection((current) => {
      if (current[resource.file_id]) {
        const next = { ...current };
        delete next[resource.file_id];
        return next;
      }
      return {
        ...current,
        [resource.file_id]: resource,
      };
    });
  };

  const confirmComposerResourceSelection = (): void => {
    if (!activeConversation) {
      return;
    }
    const selectedResources = Object.values(composerResourcePickerSelection);
    if (selectedResources.length === 0) {
      return;
    }
    stageResourcesForConversation(activeConversation.id, selectedResources);
    if (activeComposerWorkflowPreset?.clearsAfterResourcePick) {
      clearActiveComposerWorkflowPreset();
    }
    setActiveComposerResourceId(null);
    setComposerResourcePickerSelection({});
    setComposerResourceQuery("");
    setComposerResourcePickerOpen(false);
    setDismissedSlashPrompt(null);
    focusComposerTextarea();
  };

  const cancelComposerResourcePicker = (): void => {
    if (activeComposerWorkflowPreset?.clearsAfterResourcePick) {
      clearActiveComposerWorkflowPreset();
    }
    setActiveComposerResourceId(null);
    setComposerResourcePickerSelection({});
    setComposerResourceQuery("");
    setComposerResourcePickerOpen(false);
    focusComposerTextarea();
  };

  const handleSelectComposerWorkflow = (workflow: ComposerWorkflowDefinition): void => {
    if (!activeConversation) {
      return;
    }
    const preset = toComposerWorkflowPresetState(workflow);
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      updatedAt: Date.now(),
      composerWorkflowPreset: preset,
    }));
    setActivePromptValue(workflow.prompt);
    setActivePanel("chat");
    setResourceViewerContext(null);
    setDismissedSlashPrompt(null);
    const needsResourcePicker =
      workflow.opensResourcePickerOnSelect ||
      (workflow.requiresAttachedFiles && !hasComposerAttachedFiles);
    if (needsResourcePicker) {
      openComposerResourcePicker();
      return;
    }
    setActiveComposerResourceId(null);
    setComposerResourcePickerSelection({});
    setComposerResourceQuery("");
    setComposerResourcePickerOpen(false);
    focusComposerTextarea();
  };

  const handleSelectComposerIntelligenceMode = useCallback((mode: ComposerIntelligenceMode): void => {
    if (mode === "high") {
      clearActiveComposerWorkflowPreset();
      setDismissedSlashPrompt(null);
      focusComposerTextarea();
      return;
    }
    if (!activeConversation) {
      return;
    }
    const preset = toComposerWorkflowPresetState(PRO_MODE_COMPOSER_WORKFLOW_PRESET);
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      updatedAt: Date.now(),
      composerWorkflowPreset: preset,
    }));
    setActivePanel("chat");
    setResourceViewerContext(null);
    setDismissedSlashPrompt(null);
    setActiveComposerResourceId(null);
    setComposerResourcePickerSelection({});
    setComposerResourceQuery("");
    setComposerResourcePickerOpen(false);
    focusComposerTextarea();
  }, [
    activeConversation,
    clearActiveComposerWorkflowPreset,
    focusComposerTextarea,
    updateConversation,
  ]);

  const handleComposerResourceInputKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>
  ): void => {
    if (event.nativeEvent.isComposing) {
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      cancelComposerResourcePicker();
      return;
    }
    if (composerResources.length === 0) {
      if (event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
      }
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      event.stopPropagation();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      const currentIndex = composerResources.findIndex(
        (resource) => resource.file_id === resolvedActiveComposerResourceId
      );
      const nextIndex =
        currentIndex < 0
          ? 0
          : (currentIndex + direction + composerResources.length) % composerResources.length;
      setActiveComposerResourceId(composerResources[nextIndex]?.file_id ?? null);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      event.stopPropagation();
      const activeResource =
        composerResources.find((resource) => resource.file_id === resolvedActiveComposerResourceId) ??
        composerResources[0];
      if (activeResource) {
        toggleComposerResourceSelection(activeResource);
      }
    }
  };

  const useHdf5DatasetInChat = (fileId: string, datasetPaths: string[]): void => {
    if (!activeConversation) {
      return;
    }
    const uploaded = viewerUploadedFiles.find((file) => file.file_id === fileId);
    if (!uploaded) {
      return;
    }
    const bisqueLink = viewerBisqueLinksByFileId[fileId];
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      updatedAt: Date.now(),
      uploadedFiles: uniqueByFileId([...conversation.uploadedFiles, uploaded]),
      stagedUploadFileIds: uniqueFileIds([...conversation.stagedUploadFileIds, uploaded.file_id]),
      activeSelectionContext: {
        context_id: makeId(),
        source: "hdf5_dashboard",
        focused_file_ids: [uploaded.file_id],
        resource_uris: bisqueLink?.resourceUri ? [bisqueLink.resourceUri] : [],
        originating_message_id: null,
        originating_user_text:
          [...conversation.messages]
            .reverse()
            .find((message) => message.role === "user" && message.content.trim().length > 0)
            ?.content?.trim() || null,
        suggested_domain: "materials",
        suggested_tool_names: [],
      },
      bisqueLinksByFileId: bisqueLink
        ? {
            ...conversation.bisqueLinksByFileId,
            [uploaded.file_id]: bisqueLink,
          }
        : conversation.bisqueLinksByFileId,
    }));
    const uniquePaths = Array.from(new Set(datasetPaths.map((path) => String(path || "").trim()).filter(Boolean)));
    const promptSeed =
      uniquePaths.length === 1
        ? `Please analyze the HDF5 dataset \`${uniquePaths[0]}\` from the attached file.`
        : `Please analyze these HDF5 datasets from the attached file:\n${uniquePaths
            .map((path) => `- \`${path}\``)
            .join("\n")}`;
    setActivePromptValue((current) => {
      const trimmed = current.trim();
      if (uniquePaths.some((path) => trimmed.includes(path))) {
        return current;
      }
      return trimmed ? `${trimmed}\n\n${promptSeed}` : promptSeed;
    });
    setActivePanel("chat");
    setViewerOpen(false);
    setResourceViewerContext(null);
  };

  const removeDeletedResourcesFromClientState = (fileIds: string[]): void => {
    const deletedFileIds = new Set(uniqueFileIds(fileIds));
    if (deletedFileIds.size === 0) {
      return;
    }
    setResources((previous) => previous.filter((item) => !deletedFileIds.has(item.file_id)));
    setConversations((previous) =>
      previous.map((conversation) => {
        const filteredUploads = conversation.uploadedFiles.filter(
          (item) => !deletedFileIds.has(item.file_id)
        );
        const nextStagedUploadFileIds = conversation.stagedUploadFileIds.filter(
          (id) => !deletedFileIds.has(id)
        );
        if (
          filteredUploads.length === conversation.uploadedFiles.length &&
          nextStagedUploadFileIds.length === conversation.stagedUploadFileIds.length
        ) {
          return conversation;
        }
        const nextFailed = { ...conversation.failedUploadPreviewIds };
        const nextBisqueLinks = { ...conversation.bisqueLinksByFileId };
        deletedFileIds.forEach((fileId) => {
          delete nextFailed[fileId];
          delete nextBisqueLinks[fileId];
        });
        const currentFocusedFileIds =
          conversation.activeSelectionContext?.focused_file_ids ?? [];
        const nextFocusedFileIds = currentFocusedFileIds.filter(
          (fileId) => !deletedFileIds.has(fileId)
        );
        return {
          ...conversation,
          uploadedFiles: filteredUploads,
          stagedUploadFileIds: nextStagedUploadFileIds,
          activeSelectionContext:
            nextFocusedFileIds.length === currentFocusedFileIds.length
              ? conversation.activeSelectionContext
              : nextFocusedFileIds.length > 0 && conversation.activeSelectionContext
                ? {
                    ...conversation.activeSelectionContext,
                    focused_file_ids: nextFocusedFileIds,
                  }
                : null,
          failedUploadPreviewIds: nextFailed,
          bisqueLinksByFileId: nextBisqueLinks,
          updatedAt: Date.now(),
        };
      })
    );
    setResourceViewerContext((current) => {
      if (!current) {
        return current;
      }
      const nextFiles = current.uploadedFiles.filter(
        (item) => !deletedFileIds.has(item.file_id)
      );
      if (nextFiles.length === current.uploadedFiles.length) {
        return current;
      }
      if (nextFiles.length === 0) {
        setViewerOpen(false);
        return null;
      }
      const nextLinks = { ...current.bisqueLinksByFileId };
      deletedFileIds.forEach((fileId) => {
        delete nextLinks[fileId];
      });
      return {
        uploadedFiles: nextFiles,
        bisqueLinksByFileId: nextLinks,
      };
    });
  };

  const deleteResource = async (resource: ResourceRecord): Promise<void> => {
    const fileId = resource.file_id;
    setResourceDeletingById((previous) => ({ ...previous, [fileId]: true }));
    try {
      await apiClient.deleteResource(fileId);
      removeDeletedResourcesFromClientState([fileId]);
      setResourcesError(null);
    } catch (error) {
      setResourcesError(normalizeApiError(error));
    } finally {
      setResourceDeletingById((previous) => {
        const next = { ...previous };
        delete next[fileId];
        return next;
      });
    }
  };

  const restoreResource = async (resource: ResourceRecord): Promise<void> => {
    const fileId = resource.file_id;
    setResourceRestoringById((previous) => ({ ...previous, [fileId]: true }));
    try {
      const response = await restoreResourceRequest(apiClient, fileId);
      setResources((previous) => {
        const remaining = previous.filter((item) => item.file_id !== fileId);
        if (resourceStatusFilter === "deleted") {
          return remaining;
        }
        return [response.resource, ...remaining];
      });
      if (resourceStatusFilter === "deleted") {
        setResourceTotalCount((value) => Math.max(0, value - 1));
      }
      setResourceRefreshToken((value) => value + 1);
      setResourceCollectionRefreshToken((value) => value + 1);
      setResourcesError(null);
    } catch (error) {
      setResourcesError(normalizeApiError(error));
    } finally {
      setResourceRestoringById((previous) => {
        const next = { ...previous };
        delete next[fileId];
        return next;
      });
    }
  };

  const restoreResourceCollection = async (
    collection: ResourceCollectionRecord
  ): Promise<void> => {
    const collectionId = String(collection.collection_id || "").trim();
    if (!collectionId) {
      return;
    }
    setResourceCollectionRestoringById((previous) => ({
      ...previous,
      [collectionId]: true,
    }));
    try {
      const response = await restoreResourceCollectionRequest(apiClient, collectionId);
      setResourceCollections((previous) => {
        const remaining = previous.filter((item) => item.collection_id !== collectionId);
        if (resourceStatusFilter === "deleted") {
          return remaining;
        }
        return [response.collection, ...remaining];
      });
      setResourceRefreshToken((value) => value + 1);
      setResourceCollectionRefreshToken((value) => value + 1);
      setResourcesError(null);
    } catch (error) {
      setResourcesError(normalizeApiError(error));
    } finally {
      setResourceCollectionRestoringById((previous) => {
        const next = { ...previous };
        delete next[collectionId];
        return next;
      });
    }
  };

  const restoreSelectedResources = async (targets: ResourceRecord[]): Promise<void> => {
    const fileIds = uniqueFileIds(targets.map((resource) => resource.file_id));
    if (fileIds.length === 0) {
      return;
    }
    setResourceRestoringById((previous) => {
      const next = { ...previous };
      fileIds.forEach((fileId) => {
        next[fileId] = true;
      });
      return next;
    });
    try {
      const response = await restoreBulkResourcesRequest(apiClient, fileIds);
      const restoredIds = new Set(response.resources.map((resource) => resource.file_id));
      setResources((previous) => {
        const remaining = previous.filter((item) => !restoredIds.has(item.file_id));
        if (resourceStatusFilter === "deleted") {
          return remaining;
        }
        const existingIds = new Set(remaining.map((item) => item.file_id));
        const restoredResources = response.resources.filter(
          (resource) => !existingIds.has(resource.file_id)
        );
        return [...restoredResources, ...remaining];
      });
      if (resourceStatusFilter === "deleted") {
        setResourceTotalCount((value) => Math.max(0, value - restoredIds.size));
      }
      setResourceRefreshToken((value) => value + 1);
      setResourceCollectionRefreshToken((value) => value + 1);
      setResourcesError(null);
    } catch (error) {
      setResourcesError(normalizeApiError(error));
    } finally {
      setResourceRestoringById((previous) => {
        const next = { ...previous };
        fileIds.forEach((fileId) => {
          delete next[fileId];
        });
        return next;
      });
    }
  };

  const deleteSelectedResources = async (targets: ResourceRecord[]): Promise<void> => {
    const fileIds = uniqueFileIds(targets.map((resource) => resource.file_id));
    if (fileIds.length === 0) {
      return;
    }
    setResourceDeletingById((previous) => {
      const next = { ...previous };
      fileIds.forEach((fileId) => {
        next[fileId] = true;
      });
      return next;
    });
    try {
      await deleteBulkResourcesRequest(apiClient, fileIds);
      removeDeletedResourcesFromClientState(fileIds);
      setResourcesError(null);
    } catch (error) {
      setResourcesError(normalizeApiError(error));
    } finally {
      setResourceDeletingById((previous) => {
        const next = { ...previous };
        fileIds.forEach((fileId) => {
          delete next[fileId];
        });
        return next;
      });
    }
  };

  const requestResourceDelete = (resource: ResourceRecord): void => {
    setPendingResourceDelete(resource);
  };

  const requestBulkResourceDelete = (resourcesToDelete: ResourceRecord[]): void => {
    setPendingBulkResourceDelete(resourcesToDelete);
  };

  const uploadPendingFiles = async (
    conversationId: string,
    pendingFilesSnapshot: File[],
    existingUploadedFiles: UploadedFileRecord[]
  ): Promise<{
    allUploadedFiles: UploadedFileRecord[];
    newlyUploadedFiles: UploadedFileRecord[];
  }> => {
    if (pendingFilesSnapshot.length === 0) {
      return {
        allUploadedFiles: existingUploadedFiles,
        newlyUploadedFiles: [],
      };
    }
    const response = await apiClient.uploadFiles(pendingFilesSnapshot);
    const merged = uniqueByFileId([...existingUploadedFiles, ...response.uploaded]);
    updateConversation(conversationId, (conversation) => {
      const retainedFailures: Record<string, true> = {};
      const retainedBisqueLinks: Record<string, BisqueViewerLink> = {};
      merged.forEach((file) => {
        if (conversation.failedUploadPreviewIds[file.file_id]) {
          retainedFailures[file.file_id] = true;
        }
        const bisqueLink = conversation.bisqueLinksByFileId[file.file_id];
        if (bisqueLink) {
          retainedBisqueLinks[file.file_id] = bisqueLink;
        }
      });
      return {
        ...conversation,
        uploadedFiles: merged,
        stagedUploadFileIds: uniqueFileIds([
          ...conversation.stagedUploadFileIds.filter((fileId) =>
            merged.some((file) => file.file_id === fileId)
          ),
          ...response.uploaded.map((file) => file.file_id),
        ]),
        pendingFiles: [],
        failedUploadPreviewIds: retainedFailures,
        bisqueLinksByFileId: retainedBisqueLinks,
        updatedAt: Date.now(),
      };
    });
    return {
      allUploadedFiles: merged,
      newlyUploadedFiles: response.uploaded,
    };
  };

  const handleCopy = useCallback(async (value: string, feedbackKey?: string): Promise<void> => {
    if (!navigator.clipboard) {
      return;
    }
    try {
      await navigator.clipboard.writeText(value);
      if (feedbackKey) {
        setCopiedMessageId(feedbackKey);
        if (copyFeedbackTimeoutRef.current) {
          window.clearTimeout(copyFeedbackTimeoutRef.current);
        }
        copyFeedbackTimeoutRef.current = window.setTimeout(() => {
          setCopiedMessageId((current) => (current === feedbackKey ? null : current));
          copyFeedbackTimeoutRef.current = null;
        }, 1500);
      }
    } catch {
      // no-op in non-secure contexts
    }
  }, []);

  const handlePreviewError = (fileId: string): void => {
    updateActiveConversation((conversation) =>
      conversation.failedUploadPreviewIds[fileId]
        ? conversation
        : {
            ...conversation,
            failedUploadPreviewIds: {
              ...conversation.failedUploadPreviewIds,
              [fileId]: true,
            },
        }
    );
  };

  const resolveReuseDecision = (decision: ReuseDecision): void => {
    const resolver = reuseDecisionResolverRef.current;
    reuseDecisionResolverRef.current = null;
    setPendingReusePrompt(null);
    resolver?.(decision);
  };

  const promptReuseDecision = (
    suggestions: ResourceComputationSuggestion[],
    candidate: ReuseCandidateRun
  ): Promise<ReuseDecision> => {
    if (suggestions.length === 0) {
      return Promise.resolve("rerun");
    }
    if (reuseDecisionResolverRef.current) {
      reuseDecisionResolverRef.current("rerun");
      reuseDecisionResolverRef.current = null;
    }
    return new Promise((resolve) => {
      reuseDecisionResolverRef.current = resolve;
      setPendingReusePrompt({ suggestions, candidate });
    });
  };

  const loadPriorRunIntoConversation = async ({
    conversationId,
    assistantMessageId,
    candidate,
    consumedUploadFileIds,
  }: {
    conversationId: string;
    assistantMessageId: string;
    candidate: ReuseCandidateRun;
    consumedUploadFileIds: Set<string>;
  }): Promise<void> => {
    const toolLabels = candidate.toolNames
      .map((toolName) => reuseToolLabel(toolName))
      .filter((label) => label.length > 0);
    const toolLabel =
      toolLabels.length <= 1
        ? toolLabels[0] || "analysis"
        : `${toolLabels[0]} +${toolLabels.length - 1} more`;
    const requestedFileNames = Array.from(
      new Set(
        candidate.suggestions
          .map((item) => String(item.requested_file_name || "").trim())
          .filter((name) => name.length > 0)
      )
    );
    const fileLabel =
      requestedFileNames.length <= 1
        ? requestedFileNames[0] || "this file"
        : `${requestedFileNames[0]} +${requestedFileNames.length - 1} more`;
    const sourceConversation = String(candidate.conversationTitle || "").trim();
    const sourceLabel = sourceConversation
      ? `from "${sourceConversation}"`
      : `from run ${candidate.runId.slice(0, 8)}`;
    const sourceUpdatedAt = formatReuseTimestamp(candidate.conversationUpdatedAt);
    const sourceSuffix = sourceUpdatedAt ? ` (${sourceUpdatedAt})` : "";
    const reuseLead = `Loaded previous ${toolLabel} results for ${fileLabel} ${sourceLabel}${sourceSuffix}.`;

    updateConversation(conversationId, (conversation) => ({
      ...conversation,
      updatedAt: Date.now(),
      chatError: null,
      sending: true,
      streamingMessageId: null,
      stagedUploadFileIds:
        consumedUploadFileIds.size > 0
          ? conversation.stagedUploadFileIds.filter(
              (fileId) => !consumedUploadFileIds.has(fileId)
            )
          : conversation.stagedUploadFileIds,
      messages: [
        ...conversation.messages,
        {
          id: assistantMessageId,
          role: "assistant",
          content: "Loading previous results…",
          createdAt: Date.now(),
          runId: candidate.runId,
          progressEvents: [],
        },
      ],
    }));

    try {
      const payload = await apiClient.getRunResult(candidate.runId);
      if (payload.status !== "succeeded" || !payload.result) {
        throw new Error(
          `Cached run ${candidate.runId.slice(0, 8)} is ${payload.status}.`
        );
      }
      const recoveredText = payload.result.response_text?.trim() || "No response text returned.";
      const runId = String(payload.result.run_id || "").trim() || candidate.runId;
      const cachedSelection = deriveBisqueSelectionContextFromResponseMetadata({
        responseMetadata: payload.result?.metadata ?? null,
        source: "tool_result",
        originatingUserText: recoveredText,
      }).selectionContext;
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        updatedAt: Date.now(),
        sending: false,
        streamingMessageId: null,
        chatError: null,
        activeSelectionContext: mergeSelectionContexts(
          cachedSelection,
          conversation.activeSelectionContext ?? null
        ),
        messages: conversation.messages.map((item) =>
          item.id === assistantMessageId
            ? {
                ...item,
                content: `${reuseLead}\n\n${recoveredText}`,
                runId,
                durationSeconds: payload.result?.duration_seconds ?? item.durationSeconds,
                progressEvents: payload.result?.progress_events ?? item.progressEvents ?? [],
                responseMetadata: payload.result?.metadata ?? item.responseMetadata ?? null,
                liveStream: undefined,
              }
            : item
        ),
      }));
      hydrateRunDetails(conversationId, assistantMessageId, runId);
    } catch (error) {
      const message = normalizeApiError(error);
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        updatedAt: Date.now(),
        sending: false,
        streamingMessageId: null,
        chatError: null,
        messages: conversation.messages.map((item) =>
          item.id === assistantMessageId
            ? {
                ...item,
                content: `Unable to load cached results (${message}). Choose Run again to compute a fresh result.`,
                liveStream: undefined,
              }
            : item
        ),
      }));
    }
  };

  const hydrateRunArtifacts = useCallback(async (
    conversationId: string,
    assistantMessageId: string,
    runId: string
  ): Promise<void> => {
    try {
      const artifactResponse = await listRunArtifacts(apiClient, runId, 2000);
      const megasegSummaryArtifacts = artifactResponse.artifacts.filter((artifact) =>
        /megaseg_summary\.json$/i.test(String(artifact.path ?? ""))
      );
      const megasegFileSummaries = (
        await Promise.all(
          megasegSummaryArtifacts.slice(0, 12).map(async (artifact) => {
            try {
              const response = await fetch(
                apiClient.artifactDownloadUrl(runId, artifact.path),
                {
                  method: "GET",
                  credentials: "include",
                }
              );
              if (!response.ok) {
                return null;
              }
              const payload = await response.json();
              return toRecord(payload);
            } catch {
              return null;
            }
          })
        )
      ).filter((row): row is Record<string, unknown> => row !== null);
      const durableArtifacts = artifactResponse.artifacts.filter(
        (artifact) =>
          Math.max(0, Number(artifact.size_bytes) || 0) > 0 &&
          isHydratableRunArtifactVisual({
            path: artifact.path,
            mime_type: artifact.mime_type,
          })
      );
      if (durableArtifacts.length === 0) {
        return;
      }

      const selected = prioritizeHydratedImageArtifacts(durableArtifacts);

      updateConversation(conversationId, (conversation) => {
        const uploadedPreviewLookup = buildUploadedArtifactPreviewLookup(
          conversation.uploadedFiles
        );
        return {
          ...conversation,
          messages: conversation.messages.map((item) =>
            item.id === assistantMessageId
              ? {
                  ...item,
                  responseMetadata:
                    megasegFileSummaries.length > 0
                      ? {
                          ...(toRecord(item.responseMetadata) ?? {}),
                          ui_hydrated: {
                            ...(toRecord(toRecord(item.responseMetadata)?.ui_hydrated) ?? {}),
                            megaseg_file_summaries: megasegFileSummaries,
                          },
                        }
                      : item.responseMetadata,
                  runArtifacts: selected.map((artifact) => {
                    const canInlinePreview = isInlineImageArtifact(
                      artifact.path,
                      artifact.mime_type
                    );
                    const matchedUpload =
                      !canInlinePreview && isImageArtifactPath(artifact.path)
                        ? resolveUploadedArtifactPreview(artifact.path, uploadedPreviewLookup)
                        : null;
                    const downloadUrl = apiClient.artifactDownloadUrl(runId, artifact.path);
                    return {
                      path: artifact.path,
                      url: matchedUpload
                        ? apiClient.uploadPreviewUrl(matchedUpload.file_id)
                        : downloadUrl,
                      downloadUrl,
                      title: artifactDisplayName(artifact),
                      sourceName: artifactDisplayName(artifact),
                      sourcePath: String(artifact.source_path || "").trim() || undefined,
                      previewable: matchedUpload ? true : canInlinePreview,
                      linkedFileId: matchedUpload?.file_id ?? null,
                      resultGroupId:
                        String(artifact.result_group_id || "").trim() ||
                        null,
                    } satisfies RunImageArtifact;
                  }),
                }
              : item
          ),
        };
      });
    } catch (error) {
      console.warn("Artifact hydration failed", { runId, error });
      // non-blocking: keep chat response usable without artifact previews
    }
  }, [apiClient, updateConversation]);

  const hydrateRunEvents = useCallback(async (
    conversationId: string,
    assistantMessageId: string,
    runId: string
  ): Promise<void> => {
    try {
      const response = await listRunEvents(apiClient, runId, 200);
      if (!Array.isArray(response.events) || response.events.length === 0) {
        return;
      }
      const nextFingerprint = JSON.stringify(response.events);
      updateConversation(conversationId, (conversation) => {
        let changed = false;
        const messages = conversation.messages.map((item) => {
          if (item.id !== assistantMessageId) {
            return item;
          }
          const currentFingerprint = JSON.stringify(item.runEvents ?? []);
          if (currentFingerprint === nextFingerprint) {
            return item;
          }
          changed = true;
          return {
            ...item,
            runEvents: response.events,
          };
        });
        return changed
          ? {
              ...conversation,
              messages,
            }
          : conversation;
      });
    } catch {
      // non-blocking: keep chat response usable without step traces
    }
  }, [apiClient, updateConversation]);

  const hydrateRunDetails = useCallback((
    conversationId: string,
    assistantMessageId: string,
    runId: string
  ): void => {
    void hydrateRunArtifacts(conversationId, assistantMessageId, runId);
    void hydrateRunEvents(conversationId, assistantMessageId, runId);
  }, [hydrateRunArtifacts, hydrateRunEvents]);

  useEffect(() => {
    if (!conversationsHydrated) {
      return;
    }

    const targets: Array<{ conversationId: string; messageId: string; runId: string; key: string }> =
      [];

    conversations.forEach((conversation) => {
      conversation.messages.forEach((message) => {
        if (message.role !== "assistant" || !message.runId) {
          return;
        }
        const hydrationKey = `${conversation.id}:${message.id}:${message.runId}`;
        if (runArtifactHydrationsRef.current.has(hydrationKey)) {
          return;
        }
        const cards = buildToolResultCards(
          message.progressEvents ?? [],
          message.runArtifacts ?? [],
          conversation.uploadedFiles,
          (fileId) => apiClient.uploadPreviewUrl(fileId),
          {
            runId: message.runId,
            buildArtifactDownloadUrl: (runId, path) =>
              apiClient.artifactDownloadUrl(runId, path),
            responseMetadata: message.responseMetadata ?? null,
          }
        );
        if (!shouldHydrateRunArtifacts(message, cards)) {
          return;
        }
        targets.push({
          conversationId: conversation.id,
          messageId: message.id,
          runId: message.runId,
          key: hydrationKey,
        });
      });
    });

    if (targets.length === 0) {
      return;
    }

    let cancelled = false;
    const hydrateRunArtifactTargets = async (): Promise<void> => {
      for (const target of targets) {
        if (cancelled) {
          return;
        }
        runArtifactHydrationsRef.current.add(target.key);
        await hydrateRunArtifacts(target.conversationId, target.messageId, target.runId);
      }
    };

    void hydrateRunArtifactTargets();
    return () => {
      cancelled = true;
    };
  }, [apiClient, conversations, conversationsHydrated, hydrateRunArtifacts]);

  const runRecoveryTargets = useMemo(() => {
    if (authStatus !== "authenticated" || !conversationsHydrated) {
      return [] as Array<{
        conversationId: string;
        messageId: string;
        runId: string;
        afterSequence: number;
        key: string;
      }>;
    }
    const targets: Array<{
      conversationId: string;
      messageId: string;
      runId: string;
      afterSequence: number;
      key: string;
    }> = [];
    conversations.forEach((conversation) => {
      const candidate = [...conversation.messages]
        .reverse()
        .find((message) =>
          shouldRecoverRunResultMessage(message, {
            isStreamingMessage: conversation.streamingMessageId === message.id,
          })
        );
      if (!candidate?.runId) {
        return;
      }
      const key = `${conversation.id}:${candidate.id}:${candidate.runId}`;
      targets.push({
        conversationId: conversation.id,
        messageId: candidate.id,
        runId: candidate.runId,
        afterSequence: latestRunEventSequence(candidate.runEvents ?? []),
        key,
      });
    });
    return targets;
  }, [authStatus, conversations, conversationsHydrated]);

  useEffect(() => {
    if (runRecoveryTargets.length === 0) {
      return;
    }
    let cancelled = false;
    const recover = async (): Promise<void> => {
      for (const target of runRecoveryTargets) {
        if (cancelled) {
          return;
        }
        try {
          if (runStreamRecoveryControllersRef.current.has(target.key)) {
            continue;
          }
          const payload = await apiClient.getRunResult(target.runId);
          if (cancelled) {
            return;
          }
          if (payload.status === "pending" || payload.status === "running") {
            const controller = new AbortController();
            runStreamRecoveryControllersRef.current.set(target.key, controller);
            activeChatAbortControllersRef.current.set(target.conversationId, controller);
            updateConversation(target.conversationId, (conversation) => ({
              ...conversation,
              sending: true,
              chatError: null,
              streamingMessageId: target.messageId,
            }));
            void apiClient
              .resumeRunStream(target.runId, {
                afterSequence: target.afterSequence,
                signal: controller.signal,
                onRunEvent: (runEvent) => {
                  updateConversation(target.conversationId, (conversation) => ({
                    ...conversation,
                    messages: conversation.messages.map((message) =>
                      message.id === target.messageId
                        ? {
                            ...message,
                            runEvents: appendUniqueRunEvent(message.runEvents ?? [], runEvent),
                          }
                        : message
                    ),
                  }));
                },
                onToken: (delta) => {
                  updateConversation(target.conversationId, (conversation) => ({
                    ...conversation,
                    messages: conversation.messages.map((message) =>
                      message.id === target.messageId
                        ? {
                            ...message,
                            content: `${message.content ?? ""}${delta}`,
                          }
                        : message
                    ),
                  }));
                },
              })
              .then((response) => {
                if (controller.signal.aborted) {
                  return;
                }
                const recoveredText =
                  response.response_text?.trim() || "No response text returned.";
                applyGeneratedConversationTitle(target.conversationId, response);
                updateConversation(target.conversationId, (conversation) => ({
                  ...conversation,
                  sending: false,
                  chatError: null,
                  streamingMessageId:
                    conversation.streamingMessageId === target.messageId
                      ? null
                      : conversation.streamingMessageId,
                  messages: conversation.messages.map((message) =>
                    message.id === target.messageId
                      ? {
                          ...message,
                          content: recoveredText,
                          runId: response.run_id || message.runId,
                          durationSeconds:
                            response.duration_seconds ?? message.durationSeconds,
                          progressEvents:
                            response.progress_events ?? message.progressEvents ?? [],
                          responseMetadata:
                            response.metadata ?? message.responseMetadata ?? null,
                          liveStream: undefined,
                        }
                      : message
                  ),
                }));
                hydrateRunDetails(target.conversationId, target.messageId, response.run_id);
              })
              .catch((error) => {
                if (controller.signal.aborted) {
                  return;
                }
                updateConversation(target.conversationId, (conversation) => ({
                  ...conversation,
                  sending: false,
                  streamingMessageId:
                    conversation.streamingMessageId === target.messageId
                      ? null
                      : conversation.streamingMessageId,
                  chatError:
                    conversation.chatError ||
                    `Run ${target.runId.slice(0, 8)} stream recovery failed: ${normalizeApiError(error)}`,
                }));
              })
              .finally(() => {
                const activeController = runStreamRecoveryControllersRef.current.get(target.key);
                if (activeController === controller) {
                  runStreamRecoveryControllersRef.current.delete(target.key);
                }
                if (activeChatAbortControllersRef.current.get(target.conversationId) === controller) {
                  activeChatAbortControllersRef.current.delete(target.conversationId);
                }
              });
            continue;
          }
          if (payload.status !== "succeeded" || !payload.result) {
            updateConversation(target.conversationId, (conversation) => ({
              ...conversation,
              sending: false,
              streamingMessageId:
                conversation.streamingMessageId === target.messageId
                  ? null
                  : conversation.streamingMessageId,
              chatError:
                conversation.chatError ||
                `Run ${target.runId.slice(0, 8)} ended with status ${payload.status}.`,
            }));
            continue;
          }

          const recoveredText = payload.result.response_text?.trim() || "No response text returned.";
          applyGeneratedConversationTitle(target.conversationId, payload.result);
          updateConversation(target.conversationId, (conversation) => ({
            ...conversation,
            sending: false,
            chatError: null,
            streamingMessageId:
              conversation.streamingMessageId === target.messageId
                ? null
                : conversation.streamingMessageId,
            messages: conversation.messages.map((message) =>
              message.id === target.messageId
                ? {
                    ...message,
                    content: recoveredText,
                    runId: payload.result?.run_id || message.runId,
                    durationSeconds:
                      payload.result?.duration_seconds ?? message.durationSeconds,
                    progressEvents:
                      payload.result?.progress_events ?? message.progressEvents ?? [],
                    responseMetadata:
                      payload.result?.metadata ?? message.responseMetadata ?? null,
                    liveStream: undefined,
                  }
                : message
            ),
          }));
          hydrateRunDetails(target.conversationId, target.messageId, payload.result.run_id);
        } catch {
          // ignore transient recovery failures; polling will retry.
        }
      }
    };

    void recover();
    const intervalId = window.setInterval(() => {
      void recover();
    }, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [
    apiClient,
    applyGeneratedConversationTitle,
    hydrateRunDetails,
    runRecoveryTargets,
    updateConversation,
  ]);

  const setActivePromptValue = useCallback((
    nextValue: string | ((previous: string) => string)
  ): void => {
    if (!activeConversation) {
      return;
    }
    let resolvedValue = activeConversation.prompt;
    setComposerDraftsByConversationId((previous) => {
      const hasDraftOverride = Object.prototype.hasOwnProperty.call(
        previous,
        activeConversation.id
      );
      const baseValue = hasDraftOverride
        ? previous[activeConversation.id] ?? ""
        : activeConversation.prompt;
      resolvedValue =
        typeof nextValue === "function" ? nextValue(baseValue) : nextValue;
      if (hasDraftOverride && previous[activeConversation.id] === resolvedValue) {
        return previous;
      }
      return {
        ...previous,
        [activeConversation.id]: resolvedValue,
      };
    });
    if (dismissedSlashPrompt !== null && resolvedValue !== dismissedSlashPrompt) {
      setDismissedSlashPrompt(null);
    }
  }, [activeConversation, dismissedSlashPrompt]);

  const handleDeleteUserMessage = useCallback(
    (messageId: string): void => {
      updateActiveConversation((conversation) => {
        const nextMessages = removeMessageWithPairedResponse(
          conversation.messages,
          messageId
        );
        const activeStreamingId = conversation.streamingMessageId;
        const streamingRemoved =
          Boolean(activeStreamingId) &&
          !nextMessages.some((item) => item.id === activeStreamingId);
        return {
          ...conversation,
          updatedAt: Date.now(),
          messages: nextMessages,
          streamingMessageId: streamingRemoved ? null : activeStreamingId,
        };
      });
    },
    [updateActiveConversation]
  );

  const handleStreamingRenderComplete = useCallback(
    (messageId: string): void => {
      const conversationId = activeConversation?.id;
      if (!conversationId) {
        return;
      }
      updateConversation(conversationId, (conversation) => {
        const targetMessage = conversation.messages.find((message) => message.id === messageId);
        if (!targetMessage?.liveStream) {
          return conversation;
        }
        return {
          ...conversation,
          updatedAt: Date.now(),
          streamingMessageId:
            conversation.streamingMessageId === messageId
              ? null
              : conversation.streamingMessageId,
          messages: conversation.messages.map((message) =>
            message.id === messageId
              ? {
                  ...message,
                  liveStream: undefined,
                }
              : message
          ),
        };
      });
    },
    [activeConversation?.id, updateConversation]
  );

  const transcriptActions = useMemo<ConversationTranscriptActions>(
    () => ({
      onStopConversation: stopActiveConversation,
      onStreamingRenderComplete: handleStreamingRenderComplete,
      onCopy: handleCopy,
      onPromptBisqueAuthentication: promptBisqueAuthentication,
      onOpenConversationFilesInViewer: openConversationFilesInViewer,
      onImportBisqueResourcesIntoConversation: importBisqueResourcesIntoConversation,
      onCopyBisqueResourceUri: copyBisqueResourceUri,
      onEditUserMessage: setActivePromptValue,
      onDeleteUserMessage: handleDeleteUserMessage,
    }),
    [
      copyBisqueResourceUri,
      handleDeleteUserMessage,
      handleCopy,
      handleStreamingRenderComplete,
      importBisqueResourcesIntoConversation,
      openConversationFilesInViewer,
      promptBisqueAuthentication,
      setActivePromptValue,
      stopActiveConversation,
    ]
  );

  const handleSubmit = async (): Promise<void> => {
    const conversation = activeConversation;
    if (!conversation) {
      return;
    }

    const composerWorkflowPreset = conversation.composerWorkflowPreset;
    const text = activePrompt.trim();
    if (!text || conversation.sending || slashMenuOpen || composerResourcePickerOpen) {
      return;
    }
    if (conversation.selectionImportPending) {
      updateConversation(conversation.id, (current) => ({
        ...current,
        updatedAt: Date.now(),
        chatError: "Please wait for the active Use in Chat import to finish.",
      }));
      return;
    }
    const isFirstUserMessage = !conversation.messages.some(
      (message) => message.role === "user"
    );

    let importedUploadedFiles = conversation.uploadedFiles;
    let importErrorMessage: string | null = null;
    let importedUploadFileIdsForTurn: string[] = [];
    let quickPreviewFileIdsForTurn: string[] = [];
    let resolvedBisqueRowsForTurn: ToolResourceRow[] = [];
    let selectedToolNamesForTurn: string[] = [];
    let selectionContextForTurn: SelectionContext | null = conversation.activeSelectionContext ?? null;
    const bisqueUrls = extractBisqueUrls(text);
    const strippedPrompt = stripBisqueUrls(text);
    const useBisqueTargetSelectionContext = shouldUseBisqueTargetSelectionContext(text, bisqueUrls, {
      hasStagedUploads: conversation.pendingFiles.length > 0,
    });
    let promptForModel = text;
    const isBisqueImportOnly =
      bisqueUrls.length > 0 &&
      !useBisqueTargetSelectionContext &&
      strippedPrompt.length === 0 &&
      conversation.pendingFiles.length === 0;

    setViewerOpen(false);
    setResourceViewerContext(null);

    if (bisqueUrls.length > 0 && useBisqueTargetSelectionContext) {
      const targetToolNames = inferBisqueSelectionToolNames(text, {
        hasSelectionContext: true,
        hasStagedUploads: conversation.pendingFiles.length > 0,
      });
      const partitionedTargetUris = partitionBisqueUris(bisqueUrls);
      selectionContextForTurn = buildBisqueSelectionContext({
        source: "bisque_url_target",
        focusedFileIds: selectionContextForTurn?.focused_file_ids ?? [],
        resourceUris: partitionedTargetUris.resourceUris,
        datasetUris: partitionedTargetUris.datasetUris,
        originatingUserText: text,
        suggestedDomain: conversation.activeSelectionContext?.suggested_domain ?? null,
        suggestedToolNames: targetToolNames,
      });
      selectedToolNamesForTurn = Array.from(
        new Set([...selectedToolNamesForTurn, ...targetToolNames])
      );
    } else if (bisqueUrls.length > 0) {
      try {
        const importResponse = await apiClient.importBisqueResources(bisqueUrls);
        const importedBisqueCount = importResponse.uploaded.length;
        importedUploadFileIdsForTurn = importResponse.uploaded.map((file) => file.file_id);
        importedUploadedFiles = uniqueByFileId([
          ...conversation.uploadedFiles,
          ...importResponse.uploaded,
        ]);
        const failedImports = importResponse.imports.filter(
          (item) => item.status === "error"
        );
        const importedBisqueLinks: Record<string, BisqueViewerLink> = {};
        importResponse.imports.forEach((item) => {
          const fileId = item.uploaded?.file_id;
          const clientViewUrl = item.client_view_url;
          if (
            !isSuccessfulBisqueImportStatus(item.status) ||
            !fileId ||
            !clientViewUrl ||
            !clientViewUrl.trim()
          ) {
            return;
          }
          importedBisqueLinks[fileId] = {
            clientViewUrl,
            resourceUri: item.resource_uri ?? null,
            imageServiceUrl: item.image_service_url ?? null,
            inputUrl: item.input_url,
          };
        });
        importErrorMessage = (() => {
          if (failedImports.length === 0) {
            return null;
          }
          const sample = failedImports
            .slice(0, 2)
            .map((item) => {
              const detail = item.error?.trim();
              return detail
                ? `${item.input_url} (${detail})`
                : `${item.input_url} (import failed)`;
            })
            .join("; ");
          return `${failedImports.length} BisQue resource import(s) failed.${sample ? ` ${sample}` : ""}`;
        })();
        if (importedBisqueCount > 0) {
          const importedUris = importResponse.imports
            .map((item) => String(item.resource_uri ?? "").trim())
            .filter((value) => value.length > 0);
          const partitionedImportUris = partitionBisqueUris(importedUris);
          promptForModel =
            strippedPrompt.length > 0
              ? text
              : "Analyze the imported BisQue resource(s).";
          selectionContextForTurn = buildBisqueSelectionContext({
            source: "bisque_url_import",
            focusedFileIds: importResponse.uploaded.map((file) => file.file_id),
            resourceUris: partitionedImportUris.resourceUris,
            datasetUris: partitionedImportUris.datasetUris,
            originatingUserText: strippedPrompt.length > 0 ? text : null,
            suggestedDomain: conversation.activeSelectionContext?.suggested_domain ?? null,
            suggestedToolNames: [],
          });
        } else if (failedImports.length > 0) {
          promptForModel = text;
        }

        updateConversation(conversation.id, (current) => {
          const retainedFailedPreviews: Record<string, true> = {};
          const mergedBisqueLinks: Record<string, BisqueViewerLink> = {
            ...current.bisqueLinksByFileId,
            ...importedBisqueLinks,
          };
          const retainedBisqueLinks: Record<string, BisqueViewerLink> = {};
          importedUploadedFiles.forEach((file) => {
            if (current.failedUploadPreviewIds[file.file_id]) {
              retainedFailedPreviews[file.file_id] = true;
            }
            if (mergedBisqueLinks[file.file_id]) {
              retainedBisqueLinks[file.file_id] = mergedBisqueLinks[file.file_id];
            }
          });
          return {
            ...current,
            updatedAt: Date.now(),
            uploadedFiles: importedUploadedFiles,
            stagedUploadFileIds: uniqueFileIds([
              ...current.stagedUploadFileIds.filter((fileId) =>
                importedUploadedFiles.some((file) => file.file_id === fileId)
              ),
              ...importedUploadFileIdsForTurn,
            ]),
            failedUploadPreviewIds: retainedFailedPreviews,
            bisqueLinksByFileId: retainedBisqueLinks,
            chatError: importErrorMessage,
          };
        });

        if (importResponse.uploaded.length > 0) {
          setActivePanel("chat");
          setResourceViewerContext(null);
          setViewerOpen(false);
        }

        if (isBisqueImportOnly) {
          const userMessage: UiMessage = {
            id: makeId(),
            role: "user",
            content: text,
            createdAt: Date.now(),
          };
          const importedCount = importResponse.uploaded.length;
          const importSources = new Set(
            importResponse.imports
              .filter((item) => isSuccessfulBisqueImportStatus(item.status))
              .map((item) => item.download_source)
              .filter((value): value is string => Boolean(value && value.trim().length > 0))
          );
          const sourceSuffix =
            importSources.size > 0
              ? ` Download path: ${Array.from(importSources).join(", ")}.`
              : "";
          const assistantContent =
            importedCount > 0
              ? `Imported ${importedCount} BisQue resource${importedCount === 1 ? "" : "s"} into the chat context.${sourceSuffix}${importErrorMessage ? ` ${importErrorMessage}` : ""}`
              : `No BisQue resources were imported. ${importErrorMessage ?? "Check access, BISQUE_ROOT host, and resource URLs."}`;
          const assistantMessage: UiMessage = {
            id: makeId(),
            role: "assistant",
            content: assistantContent,
            createdAt: Date.now(),
          };
          updateConversation(conversation.id, (current) => ({
            ...current,
            title:
              current.messages.some((message) => message.role === "user")
                ? current.title
                : summarizeConversationTitle(promptForModel),
            updatedAt: Date.now(),
            prompt: "",
            messages: [...current.messages, userMessage, assistantMessage],
          }));
          clearComposerDraft(conversation.id);
          requestChatScrollToBottom();
          return;
        }
      } catch (error) {
        importErrorMessage = normalizeApiError(error);
        promptForModel = text;
        if (isBisqueAuthApiError(error)) {
          void promptBisqueAuthentication(importErrorMessage);
        }
        updateConversation(conversation.id, (current) => ({
          ...current,
          updatedAt: Date.now(),
          chatError: `BisQue import failed: ${importErrorMessage}`,
        }));
      }
    }

    if (bisqueUrls.length === 0) {
      const resolvedBisqueSelection = await resolveBisqueReferenceSelectionForPrompt(
        text,
        conversation
      );
      if (resolvedBisqueSelection) {
        promptForModel = resolvedBisqueSelection.promptForModel;
        importedUploadedFiles = uniqueByFileId([
          ...importedUploadedFiles,
          ...resolvedBisqueSelection.selectedUploads,
        ]);
        importedUploadFileIdsForTurn = uniqueFileIds([
          ...importedUploadFileIdsForTurn,
          ...resolvedBisqueSelection.selectedFileIds,
        ]);
        quickPreviewFileIdsForTurn = resolvedBisqueSelection.quickPreviewFileIds;
        resolvedBisqueRowsForTurn = resolvedBisqueSelection.resolvedRows;
        selectedToolNamesForTurn = resolvedBisqueSelection.selectedToolNames;
        selectionContextForTurn = resolvedBisqueSelection.selectionContext;
      }
    }

    const hasTurnScopedBisqueUploads =
      conversation.pendingFiles.length > 0 || importedUploadFileIdsForTurn.length > 0;
    if (
      shouldInferBisqueToolsForTurn(promptForModel, selectionContextForTurn, {
        hasStagedUploads: hasTurnScopedBisqueUploads,
      })
    ) {
      selectedToolNamesForTurn = Array.from(
        new Set([
          ...selectedToolNamesForTurn,
          ...(selectionContextForTurn?.suggested_tool_names ?? []),
          ...inferBisqueSelectionToolNames(promptForModel, {
            hasSelectionContext: hasBisqueSelectionContext(selectionContextForTurn),
            hasStagedUploads: hasTurnScopedBisqueUploads,
          }),
        ])
      );
    }

    const promptWorkflowIntentForTurn = inferPromptWorkflowIntent(promptForModel);
    const shouldPreferCurrentUploadedImageTarget =
      conversation.pendingFiles.some((file) => file.type.startsWith("image/")) &&
      !promptExplicitlyRequestsReuseLoad(promptForModel) &&
      (promptWorkflowIntentForTurn.asksForDepth ||
        promptWorkflowIntentForTurn.asksForSegmentation ||
        promptWorkflowIntentForTurn.asksForDetection);
    if (shouldPreferCurrentUploadedImageTarget && selectionContextForTurn) {
      selectionContextForTurn = {
        source: selectionContextForTurn.source ?? null,
        originating_message_id: null,
        originating_user_text: selectionContextForTurn.originating_user_text ?? null,
        suggested_domain: selectionContextForTurn.suggested_domain ?? null,
        suggested_tool_names: selectionContextForTurn.suggested_tool_names ?? [],
      };
    }

    const userMessage: UiMessage = {
      id: makeId(),
      role: "user",
      content: text,
      createdAt: Date.now(),
      uploadedFileNames: conversation.pendingFiles.map((file) => file.name),
    };

    const conversationId = conversation.id;
    const runIdempotencyKey = `${conversationId}:${userMessage.id}`;
    stopRequestedConversationIdsRef.current.delete(conversationId);
    activeChatAbortControllersRef.current.delete(conversationId);
    const nextMessages = [...conversation.messages, userMessage];
    const fallbackTitle = summarizeConversationTitle(promptForModel);
    clearComposerDraft(conversationId);
    updateConversation(conversationId, (current) => ({
      ...current,
      title: isFirstUserMessage ? fallbackTitle : current.title,
      updatedAt: Date.now(),
      prompt: "",
      // One-shot slash workflows clear after submit, but session-like modes
      // such as Pro Mode stay active until the user explicitly turns them off.
      composerWorkflowPreset: current.composerWorkflowPreset?.persistsAcrossTurns
        ? current.composerWorkflowPreset
        : null,
      messages: nextMessages,
      chatError: null,
      sending: true,
    }));
    requestChatScrollToBottom();

    let streamController: StreamController | null = null;
    let chatAbortController: AbortController | null = null;
    let assistantMessageId: string | null = null;
    let streamedText = "";
    let consumedUploadFileIds = new Set<string>();
    let chatRequestForRetry: Parameters<ApiClient["chat"]>[0] | null = null;
    let allUploadsForTurn: UploadedFileRecord[] = [];

    try {
      const uploadResult = await uploadPendingFiles(
        conversationId,
        conversation.pendingFiles,
        importedUploadedFiles
      );
      if (isChatStopRequested(conversationId)) {
        finalizeStoppedConversation({ conversationId, assistantMessageId, streamedText });
        return;
      }
      allUploadsForTurn = uploadResult.allUploadedFiles;
      const uploadById = new Map(allUploadsForTurn.map((file) => [file.file_id, file] as const));
      const activeSelectionFileIds = selectionContextForTurn?.focused_file_ids ?? [];
      const effectiveSelectedToolNamesForTurn = Array.from(
        new Set([
          ...(composerWorkflowPreset?.selectedToolNames ?? []),
          ...selectedToolNamesForTurn,
        ])
      );
      let currentUploadFileIds = uniqueFileIds([
        ...conversation.stagedUploadFileIds,
        ...activeSelectionFileIds,
        ...importedUploadFileIdsForTurn,
        ...uploadResult.newlyUploadedFiles.map((file) => file.file_id),
      ]).filter((fileId) => uploadById.has(fileId));
      if (effectiveSelectedToolNamesForTurn.includes("upload_to_bisque")) {
        const uploadMutationFileIds = uniqueFileIds([
          ...uploadResult.newlyUploadedFiles.map((file) => file.file_id),
          ...conversation.stagedUploadFileIds.filter(
            (fileId) => !activeSelectionFileIds.includes(fileId)
          ),
        ]).filter((fileId) => uploadById.has(fileId));
        if (uploadMutationFileIds.length > 0) {
          currentUploadFileIds = uploadMutationFileIds;
        }
      }
      consumedUploadFileIds = new Set(currentUploadFileIds);
      const currentUploads = currentUploadFileIds
        .map((fileId) => uploadById.get(fileId))
        .filter((file): file is UploadedFileRecord => Boolean(file));

      const imageUploadsForReuse = currentUploads.filter((file) => isImageLikeUploadedFile(file));
      const requestedReuseTools = inferReuseToolNames(promptForModel);
      const hasFreshImageUploadsForTurn = uploadResult.newlyUploadedFiles.some((file) =>
        isImageLikeUploadedFile(file)
      );
      const shouldAttemptReuseLookup =
        imageUploadsForReuse.length > 0 &&
        requestedReuseTools.length > 0 &&
        !promptRequestsFreshReuseComputation(promptForModel) &&
        (!hasFreshImageUploadsForTurn || promptExplicitlyRequestsReuseLoad(promptForModel));
      if (shouldAttemptReuseLookup) {
        try {
          const lookup = await apiClient.lookupResourceReuse({
            file_ids: imageUploadsForReuse.map((file) => file.file_id),
            tool_names: requestedReuseTools,
            prompt: promptForModel,
            limit_per_file_tool: 1,
          });
          const reuseCandidate = selectReuseCandidateRun(lookup.suggestions, promptForModel);
          if (reuseCandidate) {
            updateConversation(conversationId, (current) =>
              current.sending
                ? {
                    ...current,
                    sending: false,
                    streamingMessageId: null,
                  }
                : current
            );
            if (shouldAutoLoadReuseCandidate(promptForModel, reuseCandidate)) {
              const cachedAssistantMessageId = makeId();
              assistantMessageId = cachedAssistantMessageId;
              await loadPriorRunIntoConversation({
                conversationId,
                assistantMessageId: cachedAssistantMessageId,
                candidate: reuseCandidate,
                consumedUploadFileIds,
              });
              return;
            }
            const decision = await promptReuseDecision(lookup.suggestions, reuseCandidate);
            if (isChatStopRequested(conversationId)) {
              finalizeStoppedConversation({ conversationId, assistantMessageId, streamedText });
              return;
            }
            if (decision === "load") {
              const cachedAssistantMessageId = makeId();
              assistantMessageId = cachedAssistantMessageId;
              await loadPriorRunIntoConversation({
                conversationId,
                assistantMessageId: cachedAssistantMessageId,
                candidate: reuseCandidate,
                consumedUploadFileIds,
              });
              return;
            }
            updateConversation(conversationId, (current) => ({
              ...current,
              sending: true,
              chatError: null,
            }));
          }
        } catch {
          // Non-blocking: if reuse lookup fails, continue with normal run execution.
        }
      }

      if (isChatStopRequested(conversationId)) {
        finalizeStoppedConversation({ conversationId, assistantMessageId, streamedText });
        return;
      }

      const chatMessages = toChatWire(nextMessages);
      if (promptForModel !== text) {
        for (let idx = chatMessages.length - 1; idx >= 0; idx -= 1) {
          if (chatMessages[idx].role === "user") {
            chatMessages[idx] = { ...chatMessages[idx], content: promptForModel };
            break;
          }
        }
      }

      const newAssistantId = makeId();
      assistantMessageId = newAssistantId;
      const activeStream = createStreamController();
      chatAbortController = new AbortController();
      activeChatAbortControllersRef.current.set(conversationId, chatAbortController);
      streamController = activeStream;
      updateConversation(conversationId, (current) => ({
        ...current,
        updatedAt: Date.now(),
        streamingMessageId: assistantMessageId,
        messages: [
          ...current.messages,
          {
            id: newAssistantId,
            role: "assistant",
            content: "",
            createdAt: Date.now(),
            progressEvents: [],
            liveStream: activeStream.iterable,
            quickPreviewFileIds: quickPreviewFileIdsForTurn,
            resolvedBisqueResources: resolvedBisqueRowsForTurn,
          },
        ],
      }));
      const chatRequest = {
        messages: chatMessages,
        uploaded_files: [],
        file_ids: currentUploads.map((file) => file.file_id),
        conversation_id: conversationId,
        goal: promptForModel,
        selected_tool_names: effectiveSelectedToolNamesForTurn,
        selection_context: selectionContextForTurn,
        workflow_hint: composerWorkflowPreset?.workflowHint ?? null,
        reasoning_mode: "deep" as const,
        debug:
          Boolean(import.meta.env.DEV) &&
          composerWorkflowPreset?.workflowHint?.id === "pro_mode",
        idempotency_key: runIdempotencyKey,
      };
      chatRequestForRetry = chatRequest;

      const response = await apiClient.chatStream(chatRequest, {
        signal: chatAbortController.signal,
        onRunStarted: ({ runId }) => {
          if (!assistantMessageId || !runId) {
            return;
          }
          updateConversation(conversationId, (current) => ({
            ...current,
            messages: current.messages.map((item) =>
              item.id === assistantMessageId
                ? {
                    ...item,
                    runId,
                  }
                : item
            ),
          }));
        },
        onRunEvent: (runEvent) => {
          if (!assistantMessageId) {
            return;
          }
          updateConversation(conversationId, (current) => ({
            ...current,
            messages: current.messages.map((item) =>
              item.id === assistantMessageId
                ? {
                    ...item,
                    runEvents: appendUniqueRunEvent(item.runEvents ?? [], runEvent),
                  }
                : item
            ),
          }));
        },
        onToken: (delta) => {
          streamedText += delta;
          streamController?.push(delta);
        },
      });

      streamController.close();
      streamController = null;

      const assistantText =
        response.response_text?.trim() || streamedText.trim() || "No response text returned.";
      applyGeneratedConversationTitle(conversationId, response, fallbackTitle);
      const responseToolResultCards = buildToolResultCards(
        response.progress_events ?? [],
        [],
        allUploadsForTurn,
        (fileId) => apiClient.uploadPreviewUrl(fileId),
        {
          runId: response.run_id,
          buildArtifactDownloadUrl: (runId, path) =>
            apiClient.artifactDownloadUrl(runId, path),
          responseMetadata: response.metadata ?? null,
        }
      );
      const responseBisqueSelection = deriveBisqueSelectionContextFromToolCards({
        toolResultCards: responseToolResultCards,
        source: "tool_result",
        originatingUserText: promptForModel,
        suggestedDomain: selectionContextForTurn?.suggested_domain ?? null,
      });
      const responseMetadataBisqueSelection = deriveBisqueSelectionContextFromResponseMetadata({
        responseMetadata: response.metadata ?? null,
        source: "tool_result",
        originatingUserText: promptForModel,
        suggestedDomain: selectionContextForTurn?.suggested_domain ?? null,
      });
      const mergedResponseBisqueRows = mergeBisqueResourceRows([
        ...responseBisqueSelection.resolvedRows,
        ...responseMetadataBisqueSelection.resolvedRows,
      ]);
      const mergedResponseBisqueSelectionContext = mergeSelectionContexts(
        responseBisqueSelection.selectionContext,
        responseMetadataBisqueSelection.selectionContext
      );
      const clearsBisqueSelection =
        responseBisqueSelection.clearsSelection || responseMetadataBisqueSelection.clearsSelection;

      if (assistantMessageId) {
        const messageId = assistantMessageId;
        updateConversation(conversationId, (current) => ({
          ...current,
          updatedAt: Date.now(),
          chatError: null,
          sending: false,
          streamingMessageId:
            current.streamingMessageId === messageId ? null : current.streamingMessageId,
          stagedUploadFileIds:
            consumedUploadFileIds.size > 0
              ? current.stagedUploadFileIds.filter(
                  (fileId) => !consumedUploadFileIds.has(fileId)
                )
              : current.stagedUploadFileIds,
          activeSelectionContext: clearsBisqueSelection
            ? null
            : mergedResponseBisqueSelectionContext ?? current.activeSelectionContext,
          messages: current.messages.map((item) =>
            item.id === assistantMessageId
              ? {
                ...item,
                content: assistantText,
                runId: response.run_id,
                durationSeconds: response.duration_seconds ?? item.durationSeconds,
                progressEvents: response.progress_events ?? [],
                runEvents: item.runEvents ?? [],
                responseMetadata: response.metadata ?? item.responseMetadata ?? null,
                liveStream: undefined,
                resolvedBisqueResources: mergeBisqueResourceRows([
                  ...(item.resolvedBisqueResources ?? []),
                  ...mergedResponseBisqueRows,
                ]),
              }
            : item
          ),
        }));
        hydrateRunDetails(conversationId, messageId, response.run_id);
      }
    } catch (error) {
      let finalError: unknown = error;
      const initialMessage = normalizeApiError(error);
      const userStopped = isChatStopRequested(conversationId) || isAbortError(error);
      if (userStopped) {
        streamController?.close();
        streamController = null;
        finalizeStoppedConversation({ conversationId, assistantMessageId, streamedText });
        return;
      }
      const shouldRetryNonStream =
        streamedText.trim().length === 0 &&
        isTransientStreamTransportError(error, initialMessage);

      if (shouldRetryNonStream && chatRequestForRetry) {
        try {
          const fallbackResponse = await apiClient.chat(chatRequestForRetry);
          streamController?.close();
          streamController = null;

          const assistantText =
            fallbackResponse.response_text?.trim() ||
            "No response text returned.";
          applyGeneratedConversationTitle(conversationId, fallbackResponse, fallbackTitle);
          const fallbackToolResultCards = buildToolResultCards(
            fallbackResponse.progress_events ?? [],
            [],
            allUploadsForTurn,
            (fileId) => apiClient.uploadPreviewUrl(fileId),
            {
              runId: fallbackResponse.run_id,
              buildArtifactDownloadUrl: (runId, path) =>
                apiClient.artifactDownloadUrl(runId, path),
              responseMetadata: fallbackResponse.metadata ?? null,
            }
          );
          const fallbackBisqueSelection = deriveBisqueSelectionContextFromToolCards({
            toolResultCards: fallbackToolResultCards,
            source: "tool_result",
            originatingUserText: promptForModel,
            suggestedDomain: selectionContextForTurn?.suggested_domain ?? null,
          });
          const fallbackMetadataBisqueSelection = deriveBisqueSelectionContextFromResponseMetadata({
            responseMetadata: fallbackResponse.metadata ?? null,
            source: "tool_result",
            originatingUserText: promptForModel,
            suggestedDomain: selectionContextForTurn?.suggested_domain ?? null,
          });
          const mergedFallbackBisqueRows = mergeBisqueResourceRows([
            ...fallbackBisqueSelection.resolvedRows,
            ...fallbackMetadataBisqueSelection.resolvedRows,
          ]);
          const mergedFallbackBisqueSelectionContext = mergeSelectionContexts(
            fallbackBisqueSelection.selectionContext,
            fallbackMetadataBisqueSelection.selectionContext
          );
          const fallbackClearsBisqueSelection =
            fallbackBisqueSelection.clearsSelection || fallbackMetadataBisqueSelection.clearsSelection;

          if (assistantMessageId) {
            const messageId = assistantMessageId;
            updateConversation(conversationId, (current) => ({
              ...current,
              updatedAt: Date.now(),
              chatError: null,
              sending: false,
              streamingMessageId: null,
              stagedUploadFileIds:
                consumedUploadFileIds.size > 0
                  ? current.stagedUploadFileIds.filter(
                      (fileId) => !consumedUploadFileIds.has(fileId)
                    )
                  : current.stagedUploadFileIds,
              activeSelectionContext: fallbackClearsBisqueSelection
                ? null
                : mergedFallbackBisqueSelectionContext ?? current.activeSelectionContext,
              messages: current.messages.map((item) =>
                item.id === assistantMessageId
                  ? {
                      ...item,
                      content: assistantText,
                      runId: fallbackResponse.run_id,
                      durationSeconds:
                        fallbackResponse.duration_seconds ?? item.durationSeconds,
                      progressEvents: fallbackResponse.progress_events ?? [],
                      responseMetadata:
                        fallbackResponse.metadata ?? item.responseMetadata ?? null,
                      resolvedBisqueResources: mergeBisqueResourceRows([
                        ...(item.resolvedBisqueResources ?? []),
                        ...mergedFallbackBisqueRows,
                      ]),
                    }
                  : item
              ),
            }));
            hydrateRunDetails(conversationId, messageId, fallbackResponse.run_id);
            return;
          }
        } catch (fallbackError) {
          finalError = fallbackError;
        }
      }

      streamController?.fail(finalError);
      streamController = null;
      const message = normalizeApiError(finalError);
      if (assistantMessageId) {
        const partial = streamedText.trim();
        const fallbackContent = partial || `Error: ${message}`;
        updateConversation(conversationId, (current) => ({
          ...current,
          updatedAt: Date.now(),
          sending: false,
          chatError: message,
          streamingMessageId: null,
          stagedUploadFileIds:
            consumedUploadFileIds.size > 0
              ? current.stagedUploadFileIds.filter(
                  (fileId) => !consumedUploadFileIds.has(fileId)
                )
              : current.stagedUploadFileIds,
          messages: current.messages.map((item) =>
            item.id === assistantMessageId
                ? {
                    ...item,
                    content: fallbackContent,
                    liveStream: undefined,
                }
              : item
          ),
        }));
        return;
      }
      updateConversation(conversationId, (current) => {
        const withoutStreamingMessage = assistantMessageId
          ? current.messages.filter((item) => item.id !== assistantMessageId)
          : current.messages;
        return {
          ...current,
          updatedAt: Date.now(),
          sending: false,
          chatError: message,
          streamingMessageId: null,
          messages: [
            ...withoutStreamingMessage,
            {
              id: makeId(),
              role: "assistant",
              content: `Error: ${message}`,
              createdAt: Date.now(),
            },
          ],
        };
      });
    } finally {
      if (
        chatAbortController &&
        activeChatAbortControllersRef.current.get(conversationId) === chatAbortController
      ) {
        activeChatAbortControllersRef.current.delete(conversationId);
      }
      stopRequestedConversationIdsRef.current.delete(conversationId);
      updateConversation(conversationId, (current) =>
        current.sending
          ? {
              ...current,
              sending: false,
            }
          : current
      );
    }
  };

  const historyItems: HistoryItem[] = useMemo(() => {
    return [...conversations]
      .filter(shouldShowConversationInHistory)
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .map((conversation) => {
        const latestUserTurn = conversation.hydrated
          ? [...conversation.messages]
              .reverse()
              .find((message) => message.role === "user")
          : null;
        const latestMessage = conversation.hydrated
          ? conversation.messages[conversation.messages.length - 1]
          : null;
        const previewSource = conversation.hydrated
          ? latestUserTurn?.content ?? latestMessage?.content ?? ""
          : conversation.historyPreview;
        const preview = previewSource ? summarizePrompt(previewSource, 64) : "No messages yet";
        return {
          id: conversation.id,
          title:
            conversation.title && conversation.title !== "New conversation"
              ? conversation.title
              : "New conversation",
          preview,
          period: getPeriodLabel(conversation.updatedAt || conversation.createdAt),
          running: conversation.hydrated ? conversation.sending : conversation.historyRunning,
          messageCount: conversation.hydrated
            ? conversation.messages.length
            : conversation.historyMessageCount,
        };
      });
  }, [conversations]);
  const collapsedRecentItems = useMemo(() => historyItems.slice(0, 10), [historyItems]);
  const openHistoryItem = useCallback(
    (conversation: HistoryItem): void => {
      rememberActiveConversationScrollPosition();
      setActivePanel("chat");
      setActiveConversationId(conversation.id);
      setViewerOpen(false);
      setResourceViewerContext(null);
      void ensureConversationHydrated(conversation.id);
    },
    [ensureConversationHydrated, rememberActiveConversationScrollPosition]
  );
  const normalizedMobileConversationQuery = isPhoneView
    ? mobileConversationQuery.trim().toLowerCase()
    : "";
  const filteredHistoryItems = useMemo(() => {
    if (!normalizedMobileConversationQuery) {
      return historyItems;
    }
    return historyItems.filter((item) => {
      const haystack = `${item.title} ${item.preview}`.toLowerCase();
      return haystack.includes(normalizedMobileConversationQuery);
    });
  }, [historyItems, normalizedMobileConversationQuery]);

  const periodOrder: HistoryPeriod[] = [
    "Today",
    "Yesterday",
    "Last 7 days",
    "Older",
  ];
  const historyGroups = periodOrder
    .map((period) => ({
      period,
      conversations: filteredHistoryItems.filter((item) => item.period === period),
    }))
    .filter((group) => group.conversations.length > 0);
  const isMobileConversationSearchActive = normalizedMobileConversationQuery.length > 0;
  const pendingReuseCandidate = pendingReusePrompt?.candidate ?? null;
  const pendingReuseToolLabels = pendingReuseCandidate
    ? pendingReuseCandidate.toolNames
        .map((toolName) => reuseToolLabel(toolName))
        .filter((label) => label.length > 0)
    : [];
  const pendingReuseFileNames = pendingReuseCandidate
    ? Array.from(
        new Set(
          pendingReuseCandidate.suggestions
            .map((row) => String(row.requested_file_name || "").trim())
            .filter((name) => name.length > 0)
        )
      )
    : [];
  const pendingReuseSourceUpdatedAt = formatReuseTimestamp(
    pendingReuseCandidate?.conversationUpdatedAt ?? null
  );
  const pendingReuseMatchTypeLabel = pendingReuseCandidate
    ? pendingReuseCandidate.matchType === "sha256"
      ? "Exact file hash match"
      : "Filename match"
    : null;
  const pendingReuseMatchCount = pendingReuseCandidate?.suggestions.length ?? 0;
  const showAppShellBanner = shouldShowAppShellBanner(activePanel, uiErrorBanner);

  if (authStatus !== "authenticated") {
    if (authStatus === "checking" || authProvider === "workos") {
      return (
        <WorkOSRedirectScreen
          checking={authStatus === "checking"}
          loading={authSubmitting || authStatus === "checking"}
          errorMessage={authStatus === "checking" ? null : authError}
          statusMessage={authStatus === "checking" ? null : authNotice}
          onRetry={retryHostedAuth}
        />
      );
    }

    return (
      <Suspense fallback={<AuthScreenLoadingFallback />}>
        <LazyAuthScreen
          authProvider={authProvider}
          bisqueRoot={bisqueRootForAuth}
          bisqueHomeUrl={bisqueNavLinks?.home ?? undefined}
          allowGuest={authGuestEnabled}
          loading={authSubmitting}
          errorMessage={authError}
          statusMessage={authNotice}
          onAuthenticate={authenticateBisque}
          onStartHostedAuth={startHostedAuth}
          onRequestAccount={requestAccount}
        />
      </Suspense>
    );
  }

  return (
    <SidebarProvider
      className="app-shell h-dvh overflow-hidden"
      style={{ "--sidebar-width": "260px" } as CSSProperties}
    >
      <Sidebar collapsible="icon" className="app-sidebar">
        <CollapsedSidebarRail
          recentItems={collapsedRecentItems}
          activeConversationId={activeConversation?.id ?? null}
          resourcesActive={activePanel === "resources"}
          onCreateConversation={createNewConversation}
          onOpenResources={openResourcesPanel}
          onOpenRecent={openHistoryItem}
        />
        <SidebarHeader className="app-sidebar-header flex flex-row items-center justify-between gap-2 px-3 py-4">
          <Button
            type="button"
            variant="ghost"
            className="app-sidebar-brand-button min-w-0"
            onClick={createNewConversation}
            aria-label="Start a new chat"
            title="Start a new chat"
            {...mobileSidebarCloseProps}
          >
            <span className="app-sidebar-brand-mark bg-primary/10 text-primary flex size-8 items-center justify-center rounded-md">
              <BisqueMarkIcon className="size-4" />
            </span>
            <span className="app-shell-brand text-primary truncate">
              BisQue Ultra
            </span>
          </Button>
          <SidebarTrigger
            className="app-sidebar-trigger app-sidebar-header-trigger shrink-0"
            aria-label="Collapse sidebar"
            title="Collapse sidebar"
          />
        </SidebarHeader>
        <SidebarContent className="app-sidebar-content overflow-hidden pt-4">
          <div className="app-sidebar-static">
            <div className="app-sidebar-actions">
              <Button
                variant="ghost"
                className="app-new-chat-button group/new-chat mb-1 flex w-full items-center justify-between gap-2"
                onClick={createNewConversation}
                title="New chat (⌘+Shift+K)"
                aria-keyshortcuts="Control+Shift+K Meta+Shift+K"
                {...mobileSidebarCloseProps}
              >
                <span className="flex items-center gap-2">
                  <PlusIcon className="size-4" />
                  <span>New chat</span>
                </span>
                <span className="app-sidebar-shortcut-hint text-muted-foreground pointer-events-none ml-auto inline-flex items-center gap-1 text-[10px] opacity-0 transition-opacity duration-150 group-hover/new-chat:opacity-100">
                  <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                    ⌘
                  </kbd>
                  <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                    ⇧
                  </kbd>
                  <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                    K
                  </kbd>
                </span>
              </Button>
              <Button
                variant={activePanel === "resources" ? "secondary" : "ghost"}
                className="app-resource-browser-button group/resources mb-1 flex w-full items-center justify-between gap-2"
                onClick={openResourcesPanel}
                title="Resources (⌘+Shift+E)"
                aria-keyshortcuts="Control+Shift+E Meta+Shift+E"
                {...mobileSidebarCloseProps}
              >
                <span className="flex items-center gap-2">
                  <FolderOpen className="size-4" />
                  <span>Resources</span>
                </span>
                <span className="app-sidebar-shortcut-hint text-muted-foreground pointer-events-none ml-auto inline-flex items-center gap-1 text-[10px] opacity-0 transition-opacity duration-150 group-hover/resources:opacity-100">
                  <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                    ⌘
                  </kbd>
                  <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                    ⇧
                  </kbd>
                  <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                    E
                  </kbd>
                </span>
              </Button>
              <Button
                variant={activePanel === "training" ? "secondary" : "ghost"}
                className="app-resource-browser-button group/training mb-1 flex w-full items-center justify-between gap-2"
                onClick={openTrainingPanel}
                title="Training dashboard (⌘+Shift+T)"
                aria-keyshortcuts="Control+Shift+T Meta+Shift+T"
                {...mobileSidebarCloseProps}
              >
                <span className="flex items-center gap-2">
                  <Database className="size-4" />
                  <span>Training</span>
                </span>
                <span className="app-sidebar-shortcut-hint text-muted-foreground pointer-events-none ml-auto inline-flex items-center gap-1 text-[10px] opacity-0 transition-opacity duration-150 group-hover/training:opacity-100">
                  <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                    ⌘
                  </kbd>
                  <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                    ⇧
                  </kbd>
                  <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                    T
                  </kbd>
                </span>
              </Button>
              {authIsAdmin ? (
                <Button
                  variant={activePanel === "admin" ? "secondary" : "ghost"}
                  className="app-resource-browser-button mb-1 flex w-full items-center gap-2"
                  onClick={openAdminPanel}
                  {...mobileSidebarCloseProps}
                >
                  <Shield className="size-4" />
                  <span>Admin</span>
                </Button>
              ) : null}
              <Button
                variant={activePanel === "scientific-viewer" ? "secondary" : "ghost"}
                className="app-resource-browser-button group/scientific-viewer mb-1 flex w-full items-center justify-between gap-2"
                onClick={openScientificViewerPanel}
                title="Lens — scientific image viewer"
                {...mobileSidebarCloseProps}
              >
                <span className="flex items-center gap-2">
                  <LensSidebarIcon
                    active={activePanel === "scientific-viewer"}
                    data-icon="inline-start"
                    aria-hidden="true"
                  />
                  <span>Lens</span>
                </span>
              </Button>
            </div>
            <div className="app-sidebar-history-search md:hidden">
              <SidebarInput
                value={mobileConversationQuery}
                onChange={(event) => setMobileConversationQuery(event.target.value)}
                placeholder="Search chats"
                aria-label="Search chats"
              />
            </div>
            <SidebarGroup className="app-bisque-group">
              <SidebarGroupLabel>BisQue</SidebarGroupLabel>
              <SidebarMenu>
                {bisqueNavLinks ? (
                  <>
                    <SidebarMenuItem>
                      <SidebarMenuButton
                        asChild
                        className="app-bisque-link-button group/bisque-shortcut justify-between gap-2"
                      >
                        <a
                          href={bisqueNavLinks.home}
                          target="_blank"
                          rel="noreferrer"
                          title="Go to BisQue (⌘+Shift+O)"
                          aria-keyshortcuts="Control+Shift+O Meta+Shift+O"
                          {...mobileSidebarCloseProps}
                        >
                          <span className="app-bisque-link-main flex min-w-0 items-center gap-2">
                            <BisqueMarkIcon className="size-4 shrink-0" />
                            <span className="truncate">Go to BisQue</span>
                          </span>
                          <div className="app-sidebar-shortcut-hint text-muted-foreground pointer-events-none ml-auto inline-flex items-center gap-1 text-[10px] opacity-0 transition-opacity duration-150 group-hover/bisque-shortcut:opacity-100">
                            <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                              ⌘
                            </kbd>
                            <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                              ⇧
                            </kbd>
                            <kbd className="bg-muted border-border/70 inline-flex h-5 min-w-5 items-center justify-center rounded border px-1 font-medium leading-none">
                              O
                            </kbd>
                          </div>
                        </a>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton asChild className="app-bisque-link-button">
                        <a
                          href={bisqueNavLinks.images}
                          target="_blank"
                          rel="noreferrer"
                          {...mobileSidebarCloseProps}
                        >
                          <Images className="size-4" />
                          <span>
                            {formatBisqueShortcutLabel(
                              bisqueResourceCounts?.image,
                              "Image",
                              "Images"
                            )}
                          </span>
                        </a>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton asChild className="app-bisque-link-button">
                        <a
                          href={bisqueNavLinks.datasets}
                          target="_blank"
                          rel="noreferrer"
                          {...mobileSidebarCloseProps}
                        >
                          <Database className="size-4" />
                          <span>
                            {formatBisqueShortcutLabel(
                              bisqueResourceCounts?.dataset,
                              "Dataset",
                              "Datasets"
                            )}
                          </span>
                        </a>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                      <SidebarMenuButton asChild className="app-bisque-link-button">
                        <a
                          href={bisqueNavLinks.tables}
                          target="_blank"
                          rel="noreferrer"
                          {...mobileSidebarCloseProps}
                        >
                          <Table2 className="size-4" />
                          <span>
                            {formatBisqueShortcutLabel(
                              bisqueResourceCounts?.table,
                              "Table",
                              "Tables"
                            )}
                          </span>
                        </a>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  </>
                ) : (
                  <SidebarMenuItem>
                    <SidebarMenuButton className="app-bisque-link-button" disabled>
                      <ImageIcon className="size-4" />
                      <span>BisQue links unavailable</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                )}
              </SidebarMenu>
            </SidebarGroup>
          </div>

          <div className="app-sidebar-history-scroll">
            {historyGroups.length === 0 ? (
              <SidebarGroup className="app-history-group">
                <SidebarGroupLabel>
                  {isMobileConversationSearchActive ? "Search results" : "Recents"}
                </SidebarGroupLabel>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton className="app-history-button" disabled>
                      <span>
                        {isMobileConversationSearchActive ? "No chats match" : "No history yet"}
                      </span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroup>
            ) : (
              historyGroups.map((group) => (
                <SidebarGroup key={group.period} className="app-history-group">
                  <SidebarGroupLabel>
                    {group.period === "Today" ? "Recents" : group.period}
                  </SidebarGroupLabel>
                  <SidebarMenu>
                    {group.conversations.map((conversation) =>
                      pendingConversationRename?.id === conversation.id ? (
                        <SidebarMenuItem key={conversation.id} className="app-history-item">
                          <ConversationRenameEditor
                            conversation={conversation}
                            value={pendingConversationRename.title}
                            disabled={Boolean(conversationRenamingById[conversation.id])}
                            onTitleChange={updatePendingConversationRenameTitle}
                            onSubmit={submitConversationRename}
                            onCancel={cancelConversationRename}
                          />
                        </SidebarMenuItem>
                      ) : (
                        <ConversationHistoryRow
                          key={conversation.id}
                          conversation={conversation}
                          active={
                            activePanel === "chat" &&
                            conversation.id === activeConversation?.id
                          }
                          deleting={Boolean(conversationDeletingById[conversation.id])}
                          renaming={Boolean(conversationRenamingById[conversation.id])}
                          onOpen={openHistoryItem}
                          onCopyLink={copyConversationLink}
                          onCopyId={copyConversationId}
                          onRename={startConversationRename}
                          onDelete={requestConversationDelete}
                        />
                      )
                    )}
                  </SidebarMenu>
                </SidebarGroup>
              ))
            )}
            {conversationListHasMore ? (
              <div className="px-3 pb-4">
                <Button
                  type="button"
                  variant="outline"
                  className="w-full"
                  onClick={() => {
                    void loadMoreConversations();
                  }}
                  disabled={conversationListLoadingMore}
                >
                  {conversationListLoadingMore ? "Loading chats..." : "Load 25 more chats"}
                </Button>
              </div>
            ) : null}
          </div>
          <SidebarAccountSettingsButton
            authUser={authUser}
            authMode={authMode}
            authIsAdmin={authIsAdmin}
            themePreference={themePreference}
            onThemePreferenceChange={setThemePreference}
            onOpenSettings={() => {
              void loadAppSettingsDialogModule();
              setSettingsDialogOpen(true);
            }}
            onLogout={logoutBisque}
          />
        </SidebarContent>
      </Sidebar>
      {settingsDialogOpen ? (
        <Suspense fallback={null}>
          <LazyAppSettingsDialog
            open={settingsDialogOpen}
            onOpenChange={setSettingsDialogOpen}
            authUser={authUser}
            authMode={authMode}
            authIsAdmin={authIsAdmin}
            bisqueCredentialsLinked={bisqueCredentialsLinked}
            themePreference={themePreference}
            resolvedTheme={resolvedTheme}
            bisqueNavLinks={bisqueNavLinks}
            onThemePreferenceChange={setThemePreference}
            onOpenResources={openResourcesPanel}
            onOpenTraining={openTrainingPanel}
            onOpenAdmin={openAdminPanel}
            onLogout={logoutBisque}
            onUnlinkBisqueAccount={unlinkBisqueAccount}
            onLinkBisqueAccount={linkBisqueAccountFromSettings}
            loadProfile={loadCurrentUserProfile}
            saveProfile={saveCurrentUserProfile}
            loadTokenUsage={loadCurrentUserTokenUsage}
            formatError={normalizeApiError}
          />
        </Suspense>
      ) : null}

      <SidebarInset ref={setSidebarInsetNode}>
        <main className="app-main-shell flex min-h-0 flex-1 flex-col overflow-hidden">
          <div className="app-mobile-shell-bar md:hidden">
            <SidebarTrigger
              className="app-mobile-sidebar-trigger"
              aria-label="Open navigation"
              title="Open navigation"
            />
            <div className="app-mobile-shell-brand">BisQue Ultra</div>
          </div>

          {showAppShellBanner ? (
            <div className="bg-background z-10 shrink-0 px-4 pt-3">
              <SystemMessage variant="error" fill>
                {uiErrorBanner}
              </SystemMessage>
            </div>
          ) : null}

          {activePanel === "admin" ? (
            <Suspense
              fallback={
                <PanelLoadingState
                  title="Loading admin console..."
                  subtitle="Admin analytics and charts load on demand so the main chat stays fast."
                />
              }
            >
              <LazyAdminConsole
                overview={adminOverview}
                organizations={adminOrganizations}
                users={adminUsers}
                runs={adminRuns}
                issues={adminIssues}
                loadingOverview={adminLoadingOverview}
                loadingOrganizations={adminLoadingOrganizations}
                loadingUsers={adminLoadingUsers}
                loadingRuns={adminLoadingRuns}
                loadingIssues={adminLoadingIssues}
                error={adminError}
                runCancellingById={adminRunCancellingById}
                runRequeueingById={adminRunRequeueingById}
                userDeletingById={adminUserDeletingById}
                userUpdatingById={adminUserUpdatingById}
                deletingConversationKey={adminDeletingConversationKey}
                activeRunEventRunId={activeAdminRunEventRunId}
                runEventsById={adminRunEventsById}
                runEventsLoadingById={adminRunEventsLoadingById}
                runStatusFilter={adminRunStatusFilter}
                runQuery={adminRunQuery}
                userQuery={adminUserQuery}
                onRunStatusFilterChange={setAdminRunStatusFilter}
                onRunQueryChange={setAdminRunQuery}
                onUserQueryChange={setAdminUserQuery}
                onRefreshAll={refreshAdminConsole}
                onRefreshOrganizations={refreshAdminConsole}
                onRefreshUsers={refreshAdminConsole}
                onRefreshRuns={refreshAdminConsole}
                onRefreshIssues={refreshAdminConsole}
                onCreateOrganization={createAdminConsoleOrganization}
                onCreateUser={createAdminConsoleUser}
                onDeleteUser={(userId: string) => {
                  void deleteAdminConsoleUser(userId);
                }}
                onUpdateUserStatus={(userId: string, status: AdminUserStatus) => {
                  void updateAdminConsoleUserStatus(userId, status);
                }}
                onCancelRun={(runId: string) => {
                  void cancelAdminRun(runId);
                }}
                onRequeueRun={(runId: string) => {
                  void requeueAdminRun(runId);
                }}
                onDeleteConversation={(conversationId: string, userId: string) => {
                  void deleteAdminConversation(conversationId, userId);
                }}
                onInspectRunEvents={(runId: string) => {
                  void inspectAdminRunEvents(runId);
                }}
              />
            </Suspense>
          ) : activePanel === "resources" ? (
            <Suspense
              fallback={
                <PanelLoadingState
                  title="Loading resource browser..."
                  subtitle="Resource management and previews load separately from the chat shell."
                />
              }
            >
              <LazyResourceBrowser
                resources={resources}
                resourceCollections={resourceCollections}
                resourceCollectionsLoading={resourceCollectionsLoading}
                activeResourceCollection={activeResourceCollection}
                totalCount={resourceTotalCount}
                loading={resourcesLoading}
                loadingMore={resourcesLoadingMore}
                hasMore={resourceHasMore}
                error={resourcesError}
                query={resourceQuery}
                kindFilter={resourceKindFilter}
                sourceFilter={resourceSourceFilter}
                sharingFilter={resourceSharingFilter}
                statusFilter={resourceStatusFilter}
                tagFilter={resourceTagFilter}
                deletingFileIds={resourceDeletingById}
                restoringFileIds={resourceRestoringById}
                restoringCollectionIds={resourceCollectionRestoringById}
                onQueryChange={updateResourceQuery}
                onKindFilterChange={updateResourceKindFilter}
                onSourceFilterChange={updateResourceSourceFilter}
                onSharingFilterChange={updateResourceSharingFilter}
                onStatusFilterChange={updateResourceStatusFilter}
                onTagFilterChange={updateResourceTagFilter}
                onRefresh={refreshResources}
                onLoadMore={loadMoreResources}
                onUploadFiles={(files: File[], context?: ResourceUploadReselectionContext) => {
                  void uploadResourceFiles(files, context);
                }}
                uploading={resourcesUploading}
                uploadProgress={resourceUploadProgress}
                onDismissUploadProgress={dismissResourceUploadProgress}
                onPauseUploadProgress={(item: ResourceUploadProgress) => {
                  void pauseResourceUploadProgress(item);
                }}
                onCancelUploadProgress={(item: ResourceUploadProgress) => {
                  void cancelResourceUploadProgress(item);
                }}
                onOpenResource={openResourceInViewer}
                onUseInChat={addResourceToActiveConversation}
                onDeleteResource={(resource: ResourceRecord) => {
                  requestResourceDelete(resource);
                }}
                onRenameResource={renameResourceFromResources}
                onRestoreResource={(resource: ResourceRecord) => {
                  void restoreResource(resource);
                }}
                onRemoveResourceFromCollection={removeResourceFromActiveCollection}
                onDeleteSelectedResources={(selectedResources: ResourceRecord[]) => {
                  requestBulkResourceDelete(selectedResources);
                }}
                onRestoreSelectedResources={(selectedResources: ResourceRecord[]) => {
                  void restoreSelectedResources(selectedResources);
                }}
                onCreateCollectionFromSelection={createResourceFolderFromSelection}
                onAddSelectionToCollection={addResourcesToFolderFromSelection}
                onLoadResourceShareGrants={loadResourceShareGrantsFromResources}
                onCreateResourceShareGrant={createResourceShareGrantFromResources}
                onCreateBulkResourceShareGrants={createBulkResourceShareGrantsFromResources}
                onCreateResourceCollectionShareGrants={
                  createResourceCollectionShareGrantsFromResources
                }
                onRevokeResourceShareGrant={revokeResourceShareGrantFromResources}
                onOpenCollection={openResourceCollection}
                onRenameCollection={renameResourceCollectionFromResources}
                onDeleteCollection={deleteResourceCollectionFromResources}
                onRestoreCollection={(collection: ResourceCollectionRecord) => {
                  void restoreResourceCollection(collection);
                }}
                onClearActiveCollection={clearActiveResourceCollection}
                thumbnailUrlFor={(resource: ResourceRecord) =>
                  apiClient.resourceThumbnailUrl(resource.file_id)
                }
                zScrubThumbnail={{
                  // Gallery z-scrub is a transient thumbnail (never measured), so
                  // request bounded-resolution frames: the backend serves a small
                  // pyramid level for large planes, keeping rapid scrub snappy.
                  sliceUrlFor: (fileId: string, z: number) =>
                    apiClient.uploadSliceUrl(fileId, { axis: "z", z, fullResolution: false }),
                  loadZCount: (fileId: string) =>
                    apiClient.getUploadViewer(fileId).then((info) => info.axis_sizes.Z),
                }}
                downloadUrlFor={(resource: ResourceRecord) =>
                  apiClient.resourceDownloadUrl(resource.file_id)
                }
                onPushResourceToBisque={bisqueNavLinks ? pushResourceToBisque : undefined}
                onPushCollectionToBisque={bisqueNavLinks ? pushCollectionToBisque : undefined}
              />
            </Suspense>
          ) : activePanel === "training" ? (
            <Suspense
              fallback={
                <PanelLoadingState
                  title="Loading training dashboard..."
                  subtitle="Model lineage and training controls are loaded only when needed."
                />
              }
            >
              <LazyTrainingDashboard
                apiClient={apiClient}
                resources={resources}
                resourcesLoading={resourcesLoading}
                resourcesError={resourcesError}
                isAdmin={authIsAdmin}
              />
            </Suspense>
          ) : activePanel === "scientific-viewer" ? (
            <Suspense
              fallback={
                <PanelLoadingState
                  title="Loading scientific viewer..."
                  subtitle="The imaging workspace loads on demand so the main chat stays fast."
                />
              }
            >
              <LazyScientificViewerPage
                uploadedFiles={viewerUploadedFiles}
                bisqueLinksByFileId={viewerBisqueLinksByFileId}
                apiClient={apiClient}
                onUseHdf5DatasetInChat={useHdf5DatasetInChat}
              />
            </Suspense>
          ) : (
            <>
            <div className="relative min-h-0 flex-1 overflow-hidden">
              <ChatContainerRoot
                className="relative h-full min-h-0 flex-col"
              >
                <ChatAutoScroll
                  conversationId={activeConversation?.id ?? null}
                  conversationHydrated={activeConversationHydrated}
                  scrollRequestKey={chatScrollRequestKey}
                  scrollMemoryRef={conversationScrollMemoryRef}
                  scrollElementRef={activeChatScrollElementRef}
                  scrollWriteBlockRef={conversationScrollWriteBlockRef}
                />
                <ConversationTranscript
                  conversationHydrated={activeConversationHydrated}
                  messages={activeMessages}
                  blankChatTokenUsage={blankChatTokenUsage}
                  blankChatUsageLoading={blankChatUsageLoading}
                  blankChatUsageError={blankChatUsageError}
                  streamingMessageId={activeStreamingMessageId}
                  copiedMessageId={copiedMessageId}
                  uploadedFiles={activeAvailableUploadedFiles}
                  bisqueLinksByFileId={activeBisqueLinksByFileId}
                  apiClient={apiClient}
                  actions={transcriptActions}
                />
                <ChatContainerScrollAnchor />
                <div className="app-scroll-button-shell absolute bottom-4 left-1/2 z-10 flex w-full -translate-x-1/2 justify-end px-3 sm:px-5">
                  <div className="chat-width-frame flex justify-end">
                    <ScrollButton
                      aria-label="Jump to latest"
                      className="shadow-sm"
                      size="icon-sm"
                      variant="outline"
                    />
                  </div>
                </div>
              </ChatContainerRoot>
            </div>

          <div className="app-composer-shell bg-background z-10 shrink-0 px-3 pb-3 md:px-5 md:pb-5">
            <div className="chat-width-frame mx-auto">
              {activeChatError ? (
                <SystemMessage variant="error" fill className="mb-3">
                  {activeChatError}
                </SystemMessage>
              ) : null}
              <FileUpload
                onFilesAdded={(files) =>
                  updateActiveConversation((conversation) => ({
                    ...conversation,
                    pendingFiles: [...conversation.pendingFiles, ...files],
                  }))
                }
                multiple
              >
                <PromptInput
                  isLoading={activeSending || !activeConversationHydrated}
                  value={activePrompt}
                  onValueChange={(value) => setActivePromptValue(value)}
                  onSubmit={() => {
                    void handleSubmit();
                  }}
                  className="app-composer-card relative z-10 w-full"
                >
                  {slashMenuOpen ? (
                    <Suspense fallback={null}>
                      <LazyComposerSlashMenu
                        mode="workflow"
                        workflowQuery={slashWorkflowQuery}
                        activeWorkflowId={resolvedActiveSlashWorkflowId}
                        onSelectWorkflow={handleSelectComposerWorkflow}
                      />
                    </Suspense>
                  ) : null}
                  {composerResourcePickerOpen ? (
                    <Suspense fallback={null}>
                      <LazyComposerSlashMenu
                        mode="resource_picker"
                        preset={activeComposerWorkflowPreset}
                        resourceQuery={composerResourceQuery}
                        onResourceQueryChange={setComposerResourceQuery}
                        resources={composerResources}
                        resourcesLoading={composerResourcesLoading}
                        resourcesError={composerResourcesError}
                        activeResourceId={resolvedActiveComposerResourceId}
                        selectedResourceIds={selectedComposerResourceIds}
                        onResourceInputKeyDown={handleComposerResourceInputKeyDown}
                        onToggleResource={toggleComposerResourceSelection}
                        onConfirmResources={confirmComposerResourceSelection}
                        onCancelResourcePicker={cancelComposerResourcePicker}
                      />
                    </Suspense>
                  ) : null}
                  <div className="app-composer-card-body">
                    {activeSending ? (
                      <div className="composer-running">
                        <Loader
                          size="sm"
                          text={
                            activeStreamingTokenUsage
                              ? `${formatTokens(activeStreamingTokenUsage.total_tokens)} tokens`
                              : "BisQue Ultra is processing"
                          }
                        />
                      </div>
                    ) : null}
                    <PromptInputTextarea
                      ref={composerTextareaRef}
                      placeholder={activeConversationHydrated ? "Ask anything" : "Loading chat…"}
                      className="app-composer-textarea"
                      disabled={!activeConversationHydrated}
                      onKeyDown={(event) => {
                        if (
                          composerResourcePickerOpen &&
                          event.key === "Escape" &&
                          !event.nativeEvent.isComposing
                        ) {
                          event.preventDefault();
                          cancelComposerResourcePicker();
                          return;
                        }
                        if (slashMenuOpen && !event.nativeEvent.isComposing) {
                          if (
                            (event.key === "ArrowDown" || event.key === "ArrowUp") &&
                            filteredSlashWorkflows.length > 0
                          ) {
                            event.preventDefault();
                            const direction = event.key === "ArrowDown" ? 1 : -1;
	                            const currentIndex = filteredSlashWorkflows.findIndex(
	                              (workflow) => workflow.id === resolvedActiveSlashWorkflowId
	                            );
                            const nextIndex =
                              currentIndex < 0
                                ? 0
                                : (currentIndex +
                                    direction +
                                    filteredSlashWorkflows.length) %
                                  filteredSlashWorkflows.length;
                            setActiveSlashWorkflowId(
                              filteredSlashWorkflows[nextIndex]?.id ?? null
                            );
                            return;
                          }
	                          if (event.key === "Enter") {
	                            const selectedWorkflow =
	                              filteredSlashWorkflows.find(
	                                (workflow) => workflow.id === resolvedActiveSlashWorkflowId
	                              ) ?? filteredSlashWorkflows[0];
                            if (selectedWorkflow) {
                              event.preventDefault();
                              handleSelectComposerWorkflow(selectedWorkflow);
                            }
                            return;
                          }
                          if (event.key === "Escape") {
                            event.preventDefault();
                            setDismissedSlashPrompt(activePrompt);
                            setActiveSlashWorkflowId(null);
                            return;
                          }
                        }
                        if (
                          event.key === "Enter" &&
                          !event.shiftKey &&
                          !event.metaKey &&
                          !event.ctrlKey &&
                          !event.altKey &&
                          !event.nativeEvent.isComposing
                        ) {
                          event.preventDefault();
                          void handleSubmit();
                        }
                      }}
                    />

                    {pendingPreviewFiles.length > 0 ? (
                      <div className="composer-preview-section px-3 pt-2">
                        <div className="composer-preview-header">
                          <span>{`Selected files · ${pendingPreviewFiles.length}`}</span>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-6 rounded-full px-2 text-[11px]"
                            onClick={() =>
                              updateActiveConversation((conversation) => ({
                                ...conversation,
                                pendingFiles: [],
                              }))
                            }
                          >
                            Clear
                          </Button>
                        </div>
                        <div className="composer-preview-row">
                          {pendingPreviewFiles.map((file, index) => (
                            <article key={file.key} className="composer-preview-card">
                              {file.objectUrl ? (
                                <img
                                  src={file.objectUrl}
                                  alt={file.name}
                                  className="composer-preview-image"
                                />
                              ) : (
                                <div className="composer-preview-fallback">
                                  {file.isScientific ? "BIO" : "FILE"}
                                </div>
                              )}
                              <div className="composer-preview-meta">
                                <p className="composer-preview-name">{file.name}</p>
                                <p className="composer-preview-size">{file.sizeLabel}</p>
                              </div>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon-xs"
                                className="composer-preview-remove"
                                aria-label={`Remove ${file.name}`}
                                onClick={() =>
                                  updateActiveConversation((conversation) => ({
                                    ...conversation,
                                    pendingFiles: conversation.pendingFiles.filter(
                                      (_, itemIndex) => itemIndex !== index
                                    ),
                                  }))
                                }
                              >
                                <X className="size-3.5" />
                              </Button>
                            </article>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    {uploadedPreviewFiles.length > 0 ? (
                      <div className="composer-preview-section px-3 pt-2">
                        <div className="composer-preview-header">
                          <span>
                            {`${
                              activeSelectionContextFileIds.length > 0
                                ? "Active analysis context"
                                : "Uploaded context"
                            } · ${uploadedPreviewFiles.length}`}
                          </span>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="h-6 rounded-full px-2 text-[11px]"
                            onClick={() => {
                              setResourceViewerContext(null);
                              setViewerOpen(true);
                            }}
                          >
                            <ImageIcon className="mr-1 size-3.5" />
                            View
                          </Button>
                        </div>
                        <div className="composer-preview-row">
                          {uploadedPreviewFiles.map((file) => (
                            <article key={file.id} className="composer-preview-card">
                              {file.previewUrl ? (
                                <img
                                  src={file.previewUrl}
                                  alt={file.name}
                                  className="composer-preview-image"
                                  loading="lazy"
                                  onError={() => handlePreviewError(file.id)}
                                />
                              ) : (
                                <div className="composer-preview-fallback">
                                  {file.isScientific ? "BIO" : "FILE"}
                                </div>
                              )}
                              <div className="composer-preview-meta">
                                <p className="composer-preview-name">{file.name}</p>
                                <p className="composer-preview-size">{file.sizeLabel}</p>
                              </div>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon-xs"
                                className="composer-preview-remove"
                                aria-label={`Exclude ${file.name}`}
                                onClick={() =>
                                  updateActiveConversation((conversation) => {
                                    const currentSelection = conversation.activeSelectionContext;
                                    const nextFocusedFileIds = currentSelection?.focused_file_ids?.filter(
                                      (currentFileId) => currentFileId !== file.id
                                    ) ?? [];
                                    return {
                                      ...conversation,
                                      stagedUploadFileIds:
                                        conversation.stagedUploadFileIds.filter(
                                          (fileId) => fileId !== file.id
                                        ),
                                      activeSelectionContext:
                                        currentSelection && (
                                          currentSelection.focused_file_ids?.includes(file.id) ||
                                          (currentSelection.resource_uris?.length ?? 0) > 0 ||
                                          (currentSelection.dataset_uris?.length ?? 0) > 0
                                        )
                                          ? nextFocusedFileIds.length > 0
                                            ? {
                                                ...currentSelection,
                                                focused_file_ids: nextFocusedFileIds,
                                              }
                                            : null
                                          : currentSelection,
                                      updatedAt: Date.now(),
                                    };
                                  })
                                }
                              >
                                <X className="size-3.5" />
                              </Button>
                            </article>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    {activeComposerWorkflowPreset &&
                    activeComposerWorkflowPreset.id !== "pro_mode" ? (
                      <div className="flex flex-wrap items-center justify-between gap-2 px-3 pt-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge
                            data-testid="composer-workflow-chip"
                            variant="secondary"
                            className="rounded-full px-3 py-1 text-[11px]"
                          >
                            {activeComposerWorkflowPreset.label}
                          </Badge>
                          {activeComposerWorkflowPreset.requiresAttachedFiles &&
                          !hasComposerAttachedFiles ? (
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-7 rounded-full px-2.5 text-[11px]"
                              onClick={() => openComposerResourcePicker({ clearSelection: false })}
                            >
                              Choose resources
                            </Button>
                          ) : null}
                        </div>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="h-7 rounded-full px-2 text-[11px]"
                          onClick={() => {
                            if (composerResourcePickerOpen) {
                              setComposerResourcePickerSelection({});
                              setComposerResourceQuery("");
                              setComposerResourcePickerOpen(false);
                            }
                            clearActiveComposerWorkflowPreset();
                            focusComposerTextarea();
                          }}
                        >
                          <X className="mr-1 size-3.5" />
                          Clear workflow
                        </Button>
                      </div>
                    ) : null}

                    <PromptInputActions
                      className="app-composer-actions"
                    >
                      <div className="app-composer-actions-start">
                        <PromptInputAction tooltip="Attach files">
                          <FileUploadTrigger asChild>
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              aria-label="Attach files"
                              className="app-composer-icon-button composer-attach-button size-11 rounded-full sm:size-10"
                              disabled={!activeConversationHydrated}
                            >
                              <Plus size={18} />
                            </Button>
                          </FileUploadTrigger>
                        </PromptInputAction>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              data-testid="composer-intelligence-selector"
                              aria-label={`Intelligence: ${
                                activeComposerIntelligenceMode === "pro" ? "Pro" : "High"
                              }`}
                              className="app-composer-intelligence-trigger"
                              disabled={!activeConversationHydrated}
                            >
                              <span>
                                {activeComposerIntelligenceMode === "pro" ? "Pro" : "High"}
                              </span>
                              <ChevronDown data-icon="inline-end" aria-hidden="true" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent
                            align="end"
                            sideOffset={8}
                            className="app-composer-intelligence-menu"
                          >
                            <DropdownMenuLabel>Intelligence</DropdownMenuLabel>
                            <DropdownMenuGroup>
                              <DropdownMenuItem
                                data-active={
                                  activeComposerIntelligenceMode === "high" ? "true" : undefined
                                }
                                onClick={() => handleSelectComposerIntelligenceMode("high")}
                              >
                                <span>High</span>
                                {activeComposerIntelligenceMode === "high" ? (
                                  <Check data-icon="inline-end" aria-hidden="true" />
                                ) : null}
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                data-active={
                                  activeComposerIntelligenceMode === "pro" ? "true" : undefined
                                }
                                onClick={() => handleSelectComposerIntelligenceMode("pro")}
                              >
                                <span>Pro</span>
                                {activeComposerIntelligenceMode === "pro" ? (
                                  <Check data-icon="inline-end" aria-hidden="true" />
                                ) : null}
                              </DropdownMenuItem>
                            </DropdownMenuGroup>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                      <div className="app-composer-actions-end">
                        {activeSending ? (
                          <Button
                            size="icon"
                            type="button"
                            variant="destructive"
                            onClick={stopActiveConversation}
                            aria-label="Stop response"
                            title="Stop response"
                            className="app-composer-stop-button size-11 rounded-full sm:size-10"
                          >
                            <Square className="size-3.5 fill-current" />
                          </Button>
                        ) : (
                          <Button
                            size="icon"
                            type="submit"
                            disabled={!activeConversationHydrated || !activePrompt.trim() || slashMenuOpen}
                            aria-label="Send message"
                            title="Send message"
                            className="app-composer-submit-button size-11 rounded-full sm:size-10"
                          >
                            <ArrowUp size={18} />
                          </Button>
                        )}
                      </div>
                    </PromptInputActions>
                  </div>
                </PromptInput>
              </FileUpload>
            </div>
          </div>
            </>
          )}
        </main>
        {viewerOpen ? (
          <Suspense fallback={null}>
            <LazyUploadViewerSheet
              open={viewerOpen}
              onOpenChange={(open: boolean) => {
                setViewerOpen(open);
                if (!open) {
                  setResourceViewerContext(null);
                }
              }}
              uploadedFiles={viewerUploadedFiles}
              bisqueLinksByFileId={viewerBisqueLinksByFileId}
              apiClient={apiClient}
              onUseHdf5DatasetInChat={useHdf5DatasetInChat}
            />
          </Suspense>
        ) : null}
        <AlertDialog
          open={Boolean(pendingReusePrompt)}
          onOpenChange={(open) => {
            if (!open) {
              resolveReuseDecision("rerun");
            }
          }}
        >
          <AlertDialogContent
            size="default"
            portalContainer={sidebarInsetElement}
            overlayClassName="absolute inset-0"
            className="!absolute !left-1/2 !top-1/2 z-50 max-h-[calc(100%-1rem)] w-[min(calc(var(--user-chat-width)+2rem),calc(100%-1.5rem))] max-w-[calc(100%-1.5rem)] min-w-0 !-translate-x-1/2 !-translate-y-1/2 gap-5 overflow-y-auto overflow-x-hidden"
          >
            <div className="min-w-0 space-y-4">
              <div className="flex items-start gap-4">
                <AlertDialogMedia className="bg-primary/12 text-primary mb-0 shrink-0">
                  <Database className="size-7" />
                </AlertDialogMedia>
                <div className="min-w-0 space-y-2">
                  <AlertDialogTitle className="text-left">Reuse previous analysis?</AlertDialogTitle>
                  <AlertDialogDescription className="text-left break-words [overflow-wrap:anywhere]">
                    {pendingReuseCandidate
                      ? `Found a ${pendingReuseMatchTypeLabel?.toLowerCase()} for ${pendingReuseFileNames[0] ?? "this upload"}.`
                      : "Found a prior run for this upload."}
                  </AlertDialogDescription>
                </div>
              </div>
              {pendingReuseCandidate ? (
                <div className="w-full min-w-0 break-words [overflow-wrap:anywhere] rounded-md border border-border/70 bg-muted/30 px-3 py-3 text-left text-sm leading-relaxed">
                  <p className="text-foreground font-medium">{pendingReuseMatchTypeLabel}</p>
                  <p className="text-muted-foreground">{`Matched records: ${pendingReuseMatchCount}`}</p>
                  <p className="text-muted-foreground break-all">{`Matched file(s): ${pendingReuseFileNames.join(", ") || "unknown"}`}</p>
                  <p className="text-muted-foreground">{`Tools in prior run: ${pendingReuseToolLabels.join(", ") || "analysis"}`}</p>
                  <p className="text-muted-foreground break-all">{`Run ID: ${pendingReuseCandidate.runId}`}</p>
                  {pendingReuseCandidate.conversationTitle ? (
                    <p className="text-muted-foreground break-words [overflow-wrap:anywhere]">{`Conversation: ${pendingReuseCandidate.conversationTitle}`}</p>
                  ) : null}
                  {pendingReuseSourceUpdatedAt ? (
                    <p className="text-muted-foreground">{`Last updated: ${pendingReuseSourceUpdatedAt}`}</p>
                  ) : null}
                  <p className="text-muted-foreground pt-1">
                    Load existing outputs to continue quickly, or run again to generate fresh results.
                  </p>
                </div>
              ) : null}
            </div>
            <div className="flex w-full flex-wrap items-center justify-center gap-2">
              <AlertDialogCancel
                onClick={() => {
                  resolveReuseDecision("rerun");
                }}
              >
                Run again
              </AlertDialogCancel>
              <AlertDialogAction
                disabled={!pendingReuseCandidate}
                onClick={() => {
                  resolveReuseDecision("load");
                }}
              >
                Load results
              </AlertDialogAction>
            </div>
          </AlertDialogContent>
        </AlertDialog>
        <AlertDialog
          open={Boolean(pendingConversationDelete)}
          onOpenChange={(open) => {
            if (!open) {
              setPendingConversationDelete(null);
            }
          }}
        >
          <AlertDialogContent size="default">
            <AlertDialogHeader>
              <AlertDialogMedia className="bg-destructive/12 text-destructive">
                <Trash className="size-7" />
              </AlertDialogMedia>
              <AlertDialogTitle>Delete conversation?</AlertDialogTitle>
              <AlertDialogDescription>
                {`Delete "${pendingConversationDelete?.title ?? "this conversation"}" and remove its messages from storage?`}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel
                onClick={() => {
                  setPendingConversationDelete(null);
                }}
              >
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                variant="destructive"
                disabled={
                  !pendingConversationDelete ||
                  Boolean(conversationDeletingById[pendingConversationDelete.id])
                }
                onClick={() => {
                  if (!pendingConversationDelete) {
                    return;
                  }
                  const targetId = pendingConversationDelete.id;
                  setPendingConversationDelete(null);
                  void deleteConversationFromHistory(targetId);
                }}
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
        <AlertDialog
          open={Boolean(pendingResourceDelete)}
          onOpenChange={(open) => {
            if (!open) {
              setPendingResourceDelete(null);
            }
          }}
        >
          <AlertDialogContent size="default">
            <AlertDialogHeader>
              <AlertDialogMedia className="bg-destructive/12 text-destructive">
                <Trash className="size-7" />
              </AlertDialogMedia>
              <AlertDialogTitle>Move resource to trash?</AlertDialogTitle>
              <AlertDialogDescription>
                {`Move "${pendingResourceDelete?.original_name ?? "this file"}" to Trash. You can restore it from Deleted when needed.`}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel
                onClick={() => {
                  setPendingResourceDelete(null);
                }}
              >
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                variant="destructive"
                disabled={
                  !pendingResourceDelete ||
                  Boolean(resourceDeletingById[pendingResourceDelete.file_id])
                }
                onClick={() => {
                  if (!pendingResourceDelete) {
                    return;
                  }
                  const target = pendingResourceDelete;
                  setPendingResourceDelete(null);
                  void deleteResource(target);
                }}
              >
                Move to trash
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
        <AlertDialog
          open={pendingBulkResourceDelete.length > 0}
          onOpenChange={(open) => {
            if (!open) {
              setPendingBulkResourceDelete([]);
            }
          }}
        >
          <AlertDialogContent size="default">
            <AlertDialogHeader>
              <AlertDialogMedia className="bg-destructive/12 text-destructive">
                <Trash className="size-7" />
              </AlertDialogMedia>
              <AlertDialogTitle>Move selected resources to trash?</AlertDialogTitle>
              <AlertDialogDescription>
                {`Move ${pendingBulkResourceDelete.length.toLocaleString()} selected ${
                  pendingBulkResourceDelete.length === 1 ? "resource" : "resources"
                } to Trash. You can restore them from Deleted when needed.`}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel
                onClick={() => {
                  setPendingBulkResourceDelete([]);
                }}
              >
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                variant="destructive"
                disabled={
                  pendingBulkResourceDelete.length === 0 ||
                  pendingBulkResourceDelete.some((resource) =>
                    Boolean(resourceDeletingById[resource.file_id])
                  )
                }
                onClick={() => {
                  const targets = pendingBulkResourceDelete;
                  setPendingBulkResourceDelete([]);
                  void deleteSelectedResources(targets);
                }}
              >
                Move to trash
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </SidebarInset>
      <DeferredToaster theme={resolvedTheme} />
      {activePanel !== "resources" ? (
        <UploadFlightChip inFlightCount={inFlightUploadCount} onOpen={openResourcesPanel} />
      ) : null}
    </SidebarProvider>
  );
}
