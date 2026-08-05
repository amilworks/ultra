import {
  Suspense,
  memo,
  type CSSProperties,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal, flushSync } from "react-dom";
import {
  ChatContainerContent,
  ChatContainerRoot,
  ChatContainerScrollAnchor,
  FileUpload,
  useFileUploadContext,
  Loader,
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
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  mobileSidebarCloseProps,
} from "@/components/ui/sidebar";
import { useBreakpoint } from "@/hooks/use-breakpoint";
import { cn } from "@/lib/utils";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { lazyNamedWithRetry } from "@/lib/lazy-retry";
import {
  ApiClient,
  ApiError,
  isSteeringClosedError,
  UploadPausedError,
  type StreamTokenEvent,
  type UploadProgressEvent,
} from "./lib/api";
import { buildNavUrl, navStateKey, parseNavFromSearch, type NavState } from "./lib/navUrl";
import { FigureLightboxRoot } from "./components/FigureLightboxRoot";
import { AnimatedTokenCount } from "./components/chat/AnimatedTokenCount";
import { FigureCaption } from "./components/chat/FigureCaption";
import {
  openFigureLightbox,
  registerLightboxOpenInLens,
  type LightboxFigure,
} from "./lib/figureLightbox";
import { bundleRootForRelativePath, groupPendingUploads } from "./lib/pendingBundles";
import { filesFromClipboard } from "./lib/clipboardFiles";
import {
  draftWithQuotedSelection,
  pastedTextFile,
  shouldAttachPastedText,
} from "./lib/pasted-text";
import {
  applyTranscriptFindHighlights,
  clearTranscriptFindHighlights,
  computeTranscriptFindMatches,
} from "./lib/transcript-find";
import { textFromSelection } from "./lib/selection-capture";
import { TranscriptFindBar } from "./components/chat/TranscriptFindBar";
// Type-only: erased at compile time, so react-virtuoso itself stays lazy.
import type { VirtuosoHandle } from "react-virtuoso";
import {
  collectDroppedFiles,
  isOsFileDrag,
  MAX_DROPPED_FILES,
  snapshotDropPayload,
  summarizeDropIssues,
} from "./lib/dropTraversal";
import {
  DEFAULT_API_BASE_URL,
  DEFAULT_API_KEY,
  DEFAULT_BISQUE_BROWSER_URL,
} from "./lib/config";
import { buildBisqueThumbnailUrl } from "./lib/bisquePreview";
import { remoteMutationIntentsForUserText } from "./lib/bisqueMutationIntent";
import { formatBytes, formatTokens } from "./lib/format";
import {
  thumbnailScrubConfig,
  thumbnailScrubSliceRequest,
  type ThumbnailScrubConfig,
} from "./lib/thumbnailScrubAxis";
import {
  buildBisqueNavLinks,
  inferBisqueRootFromUrl,
  type BisqueNavLinks,
} from "./features/auth/bisqueNavigation";
import {
  createAdminOrganization,
  createAdminUser,
  deleteAdminUser,
  loadAdminMetrics,
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
  classifyRunDocumentKind,
  isHydratableRunArtifactDocument,
  isHydratableRunArtifactVisual,
  rewriteArtifactMarkdownImageUrls,
  type RunDocumentKind,
  runReportPathKey,
  shouldHydrateRunArtifacts,
} from "./features/chat/run-artifact-hydration";
import type {
  ReportCanvasMode,
  ReportCanvasVersion,
} from "./components/canvas/ReportCanvas";
import {
  isLegacyRunLookup404Content,
  latestRunEventSequence,
  shouldDropHydratedLegacy404Message,
  shouldRecoverRunResultMessage,
} from "./features/chat/run-recovery";
import { extractRunTokenUsage } from "./features/chat/token-usage";
import {
  appendRunEventCoalescing,
  isEphemeralDeltaEvent,
  reasoningTextFromRunEvents,
  runHasToolActivity,
} from "./features/chat/run-events";
import { MESSAGE_WINDOW_SIZE, windowTailMessages } from "./features/chat/message-window";
import { isTabHidden, onVisibilityChange } from "./features/chat/tab-visibility";
import {
  classifyStreamFailure,
  composeStreamFailureReason,
} from "./features/chat/stream-failure";
import {
  findReusableBlankDraftConversation,
  shouldExposeConversationInUrl,
  shouldShowConversationInHistory,
  shouldPersistConversationSnapshot,
} from "./features/chat/conversation-draft";
import {
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
import { DEFAULT_THINKING_TEXT } from "./lib/runStepCopy";
import { useLocalStorageState } from "./lib/useLocalStorageState";
import { UserTokenUsagePanel } from "./components/UserTokenUsagePanel";
import type {
  AdminCreateOrganizationRequest,
  AdminCreateUserRequest,
  AdminIssueRecord,
  AdminOrganization,
  AdminMetricsResponse,
  AdminOverviewResponse,
  AdminRunRecord,
  AdminUserStatus,
  AdminUserSummary,
  ArtifactRecord,
  BisqueAuthSessionResponse,
  ChatResponse,
  ChatMessage,
  ConversationRecord,
  CurrentUserProfile,
  ProgressEvent,
  ResourceCollectionRecord,
  ResourceRecord,
  ResourceShareGrantRecord,
  ResourceCollectionShareGrantRecord,
  ShareTargetRecord,
  RunEvent,
  SelectionContext,
  TokenUsageResponse,
  UploadedFileRecord,
} from "./types";
import type { SettingsTab } from "./components/AppSettingsDialog";
import { BrandWordmark } from "./components/BrandWordmark";
import { BisqueMarkIcon } from "./components/icons/BisqueMarkIcon";
import { LensSidebarIcon } from "./components/icons/LensSidebarIcon";
import { LiveStreamRegion } from "./components/chat/LiveStreamRegion";
import { ReasoningTrace } from "./components/chat/ReasoningTrace";
import {
  composeComposerWorkflowPromptForModel,
  slashWorkflowSearchQuery,
  visiblePromptAfterComposerWorkflowSelection,
} from "./components/chat/composer-workflow-prompt";
import type {
  ComposerWorkflowDefinition,
  ComposerWorkflowId,
  ComposerWorkflowPresetState,
} from "./components/chat/composer-workflows";
import type { ComposerWorkflowGroup } from "./components/chat/ComposerSlashMenu";
import type {
  PrairieImageAnalysis,
  ToolCardImage,
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
  FileUp,
  FolderUp,
  History,
  ImageIcon,
  Images,
  Layers,
  Link2,
  Pencil,
  Plus,
  PlusIcon,
  Search,
  Shield,
  Square,
  SquarePen,
  Table2,
  TextQuote,
  Trash,
  X,
  Zap,
} from "lucide-react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { toNumber, toRecord } from "./lib/coerce";
import { queueEffectUpdate } from "./lib/queueEffectUpdate";
import { showErrorToast, showSuccessToast, showUndoToast } from "./lib/toast";
import { useThemePreference } from "./lib/useThemePreference";
import {
  MISSING_REQUESTED_CONVERSATION_MESSAGE,
  shouldShowAppShellBanner,
} from "./features/app-shell-banner";
import { dedupeBisqueResourceRows } from "./features/chat/bisque-resource-rows";
import {
  HISTORY_PERIOD_ORDER,
  type HistoryItem,
  type HistoryPeriod,
} from "./features/chat/history";
import { useBlankChatTokenUsage } from "./features/chat/useBlankChatTokenUsage";
import { DeferredToaster } from "./components/DeferredToaster";
import { PanelLoadingState } from "./components/PanelLoadingState";
import { UploadFlightChip } from "./components/UploadFlightChip";
import {
  AuthScreenLoadingFallback,
  WorkOSRedirectScreen,
} from "./components/auth/AuthShellScreens";
import { AssistantTurnRecovery } from "./components/chat/AssistantTurnRecovery";
import {
  ChatAutoScroll,
  captureConversationScrollMemory,
  type ConversationScrollMemory,
} from "./components/chat/ChatAutoScroll";
import { CollapsedSidebarRail } from "./components/chat/CollapsedSidebarRail";
import {
  ConversationHistoryRow,
  ConversationRenameEditor,
} from "./components/chat/ConversationHistoryRow";
import { SidebarAccountSettingsButton } from "./components/chat/SidebarAccountSettingsButton";

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

// Viewerinfo is resolved lazily on first hover. Keep its complete display contract
// per file so every synchronous scrub URL preserves the authoritative composite,
// LUT palette, fixed time/depth, and selected scrub axis.
const THUMBNAIL_SCRUB_CONFIG = new Map<string, ThumbnailScrubConfig>();

/* Report canvas split geometry, all in px of the MAIN SHELL's width.
   Split needs transcript-min + panel-min to coexist; below that the canvas
   is a sheet. The default matches the pre-resize fixed column (40.5rem). */
const REPORT_CANVAS_TRANSCRIPT_MIN = 384;
const REPORT_CANVAS_PANEL_MIN = 320;
const REPORT_CANVAS_PANEL_MAX = 896;
const REPORT_CANVAS_PANEL_DEFAULT = 648;
const REPORT_CANVAS_SPLIT_MIN_STAGE =
  REPORT_CANVAS_TRANSCRIPT_MIN + REPORT_CANVAS_PANEL_MIN + 16;
const REPORT_CANVAS_WIDTH_STORAGE_KEY = "ultra:report-canvas:width";

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

// A non-visual durable run output (markdown report, supporting code, data table)
// surfaced in the chat for inline reading and download. Images use RunImageArtifact.
type RunDocumentArtifact = {
  path: string;
  title: string;
  downloadUrl: string;
  kind: RunDocumentKind;
  mimeType?: string;
  sizeBytes?: number;
};

// Fetch a durable artifact's text for inline rendering. The download URL is
// absolute and same-origin-authenticated, so credentials must ride along.
const fetchRunDocumentText = async (downloadUrl: string): Promise<string> => {
  const response = await fetch(downloadUrl, { method: "GET", credentials: "include" });
  if (!response.ok) {
    throw new Error(`Unable to load document (${response.status})`);
  }
  return response.text();
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

// Hardened against ChunkLoadError: retries transient import failures and
// recovers from stale-deploy chunk-hash rotation with a single guarded reload
// rather than crashing the whole app via the top-level boundary.
const lazyNamed = lazyNamedWithRetry;

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
const loadToolResultCardsModule = () => import("./components/chat/ToolResultCards");
const loadChatRunDocumentsModule = () => import("./components/chat/ChatRunDocuments");
const loadReportCanvasModule = () => import("./components/canvas/ReportCanvas");
const loadComposerWorkflowsModule = () => import("./components/chat/composer-workflows");
const loadVirtuosoModule = () => import("react-virtuoso");

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
const LazyVirtuoso = lazyNamed(loadVirtuosoModule, "Virtuoso");
const LazyToolResultCardSection = lazyNamed(
  loadToolResultCardsModule,
  "ToolResultCardSection"
);
const LazyChatRunDocuments = lazyNamed(loadChatRunDocumentsModule, "ChatRunDocuments");
const LazyReportCanvas = lazyNamed(loadReportCanvasModule, "ReportCanvas");

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

type UiMessageStatus = "stopped" | "failed";

/* Mid-run steering lifecycle (Phase 1 of double texting). pending = accepted
   by the control plane, waiting for the worker's next model-call boundary;
   applied = the running agent saw it; missed = the run ended first (the
   message stays in the transcript, so the NEXT turn reads it); historic =
   hydrated from a past conversation, lifecycle over. */
type UiSteeringStatus = "pending" | "applied" | "missed" | "historic";

type UiMessage = {
  id: string;
  role: UiRole;
  content: string;
  createdAt: number;
  // A turn that the user stopped, or that failed to complete. Drives the calm
  // inline recovery affordance (Retry / Edit) instead of a silent dead-end.
  status?: UiMessageStatus;
  // Present only on user messages sent as mid-run steering.
  steering?: UiSteeringStatus;
  steerId?: string;
  // Raw technical detail for a failed turn, shown in muted monospace.
  errorReason?: string;
  runId?: string;
  durationSeconds?: number;
  progressEvents?: ProgressEvent[];
  runEvents?: RunEvent[];
  // The coordinator's accumulated thinking trace (chain-of-thought), surfaced under the collapsible
  // "Thinking" expansion. Derived from the coalesced trace.reasoning.delta run event and persisted
  // separately so it survives the turn (the reasoning deltas themselves are stripped from snapshots).
  reasoning?: string;
  responseMetadata?: Record<string, unknown> | null;
  uploadedFileNames?: string[];
  liveStream?: AsyncIterable<string>;
  runArtifacts?: RunImageArtifact[];
  runDocuments?: RunDocumentArtifact[];
  quickPreviewFileIds?: string[];
  resolvedBisqueResources?: ToolResourceRow[];
};

type BisqueViewerLink = {
  clientViewUrl: string;
  resourceUri?: string | null;
  imageServiceUrl?: string | null;
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
  /* Runs the user deliberately removed. Persisted with the snapshot so hydration
     stops rebuilding a deleted turn from control_runs.response_text — see
     readDeletedRunIds in lib/api.ts for why the absence alone is not enough. */
  deletedRunIds: string[];
  /* Phase-0 follow-ups ("double texting", enqueue flavour): text composed while
     a run is in flight, dispatched as the NEXT turn on clean completion. One
     growing message, not a list — additional mid-run sends append a paragraph,
     so completion fires exactly one run instead of a surprise chain of them. */
  queuedFollowup: string;
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

// Fold a run event into a message: append to runEvents (coalescing reasoning) AND capture the
// accumulated reasoning stickily on message.reasoning. Reasoning must be captured as it streams
// because the post-completion runEvents replacement (final server events are pruned/ephemeral) would
// otherwise drop it — leaving the "Thought process" expansion nothing durable to read.
const foldRunEventIntoMessage = <M extends { runEvents?: RunEvent[]; reasoning?: string }>(
  message: M,
  runEvent: RunEvent
): M => {
  const runEvents = appendRunEventCoalescing(message.runEvents ?? [], runEvent);
  return {
    ...message,
    runEvents,
    reasoning: reasoningTextFromRunEvents(runEvents) || message.reasoning,
  };
};
const EMPTY_RUN_IMAGE_ARTIFACTS: RunImageArtifact[] = [];
const EMPTY_RUN_DOCUMENTS: RunDocumentArtifact[] = [];
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

// Map run-output image artifacts to the figure-lightbox shape (previewable only).
const runArtifactsToFigures = (artifacts: RunImageArtifact[]): LightboxFigure[] =>
  artifacts
    .filter((artifact) => artifact.previewable)
    .map((artifact) => ({
      url: artifact.url,
      downloadUrl: artifact.downloadUrl,
      title: artifact.title,
      fileId: artifact.linkedFileId ?? undefined,
    }));

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

const CONVERSATION_QUERY_PARAM = "conversation";
const CONVERSATION_PAGE_SIZE = 25;
const RESOURCE_PAGE_SIZE = 50;

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

type ResourceListRequestParams = {
  collectionId: string;
  query: string;
  kind: ResourceKindFilter;
  source: ResourceSourceFilter;
  sharing: ResourceSharingFilter;
  status: ResourceStatusFilter;
  tags: string[];
  refreshToken: number;
};

// Fingerprint + request construction shared by the resource-list effect and
// loadMoreResources, so a one-sided filter addition can't silently break the
// stale-response guard.
const buildResourceListKey = (params: ResourceListRequestParams): string =>
  [
    params.collectionId ? `folder:${params.collectionId}` : "library",
    params.query,
    params.kind,
    params.source,
    params.sharing,
    params.status,
    params.tags.join("\u0001"),
    String(params.refreshToken),
  ].join("\u0000");

const buildResourceListRequest = (
  apiClient: ApiClient,
  params: ResourceListRequestParams,
  offset: number
) =>
  params.collectionId
    ? loadResourceFolderResources(apiClient, params.collectionId, {
        limit: RESOURCE_PAGE_SIZE,
        offset,
        query: params.query || undefined,
        kind: params.kind,
        source: params.source,
        sharing: params.sharing,
        status: params.status,
        tags: params.tags,
      })
    : loadLibraryResources(apiClient, {
        limit: RESOURCE_PAGE_SIZE,
        offset,
        query: params.query || undefined,
        kind: params.kind,
        source: params.source,
        sharing: params.sharing,
        status: params.status,
        tags: params.tags,
      });

// Shared ArrowUp/ArrowDown wrap-around cycling for the slash menu and the
// composer resource picker.
const cycleListIndex = (
  currentIndex: number,
  direction: number,
  length: number
): number => (currentIndex < 0 ? 0 : (currentIndex + direction + length) % length);

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

// Switching threads is a navigation: PUSH a history entry so Back returns to the
// previous conversation (the convention every mainstream chat product follows).
// Programmatic normalization (first sync, clearing a draft) still replaces.
const pushConversationIdInLocation = (conversationId: string): void => {
  if (typeof window === "undefined") {
    return;
  }
  const normalizedConversationId = conversationId.trim();
  if (!normalizedConversationId) {
    return;
  }
  const nextUrl = new URL(window.location.href);
  nextUrl.searchParams.set(CONVERSATION_QUERY_PARAM, normalizedConversationId);
  nextUrl.searchParams.delete("auth_error");
  const nextRelativeUrl = `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`;
  const currentRelativeUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (nextRelativeUrl === currentRelativeUrl) {
    return;
  }
  window.history.pushState({}, "", nextRelativeUrl);
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

type BisqueReferenceSelection = {
  sourceRows: ToolResourceRow[];
  selectedRows: ToolResourceRow[];
  intent: "preview" | "selection";
};

type BisqueImportedSelection = {
  uploadedFiles: UploadedFileRecord[];
  bisqueLinksByFileId: Record<string, BisqueViewerLink>;
};

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

/**
 * Should this keystroke pull focus into the composer?
 *
 * The pain being fixed: you look at a chat, start typing your next prompt, and
 * discover several words later that nothing was focused and the text went
 * nowhere. Slack, Linear and Discord all solve it the same way — any ordinary
 * character types into the message box no matter where focus happens to be.
 *
 * Everything here is about NOT stealing a keystroke that meant something else:
 *
 * - `key.length === 1` admits printable characters and nothing else. Enter,
 *   Tab, Escape, Backspace, arrows and F-keys all report longer names, so
 *   navigation and shortcuts are untouched.
 * - Space is deliberately excluded even though it is printable: it scrolls the
 *   transcript, and a prompt essentially never begins with a space.
 * - Any Meta/Ctrl/Alt combination belongs to a shortcut (⌘K, ⌘C, ⌘⇧E…), so it
 *   passes straight through. Plain Shift is allowed — capitals start sentences.
 * - `isComposing` protects IME input; hijacking focus mid-composition would
 *   destroy in-progress CJK text.
 */
const shouldTypingFocusComposer = (event: KeyboardEvent): boolean => {
  if (event.metaKey || event.ctrlKey || event.altKey) {
    return false;
  }
  if (event.isComposing || event.defaultPrevented) {
    return false;
  }
  if (event.key.length !== 1 || event.key === " ") {
    return false;
  }
  return true;
};

/**
 * A dialog, sheet or popover is open, so the transcript is not what the user is
 * typing into. Radix marks its open overlays with `data-state="open"`, and locks
 * the body while a modal is up; either signal is enough to stand down.
 */
const hasBlockingOverlay = (): boolean =>
  Boolean(
    document.querySelector(
      '[role="dialog"][data-state="open"], [role="alertdialog"][data-state="open"], [role="menu"][data-state="open"], [data-slot="popover-content"][data-state="open"]'
    )
  ) || document.body.hasAttribute("data-scroll-locked");

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

  return {
    asksForSegmentation,
    asksForDepth,
    asksForDetection,
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
    deletedRunIds: [],
    queuedFollowup: "",
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
        status:
          row.status === "stopped" || row.status === "failed"
            ? (row.status as UiMessageStatus)
            : undefined,
        errorReason:
          typeof row.errorReason === "string" && row.errorReason.trim()
            ? row.errorReason
            : undefined,
        runId: row.runId ? String(row.runId) : undefined,
        steering:
          row.steering === "pending" ||
          row.steering === "applied" ||
          row.steering === "missed" ||
          row.steering === "historic"
            ? (row.steering as UiSteeringStatus)
            : undefined,
        steerId:
          typeof row.steerId === "string" && row.steerId.trim() ? row.steerId : undefined,
        durationSeconds: toNumber(row.durationSeconds ?? row.duration_seconds) ?? undefined,
        progressEvents: toProgressEvents(row.progressEvents),
        runEvents: toRunEvents(row.runEvents),
        reasoning:
          typeof row.reasoning === "string" && row.reasoning.trim()
            ? row.reasoning
            : undefined,
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
      // Steering messages belong to the turn being removed: they sit between
      // the originating prompt and its assistant, and orphaning them would
      // strand "Steered mid-run" bubbles with no turn to steer.
      if (candidate.role === "user" && candidate.steering) {
        idsToRemove.add(candidate.id);
        continue;
      }
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
    /* Read the tombstones back, or the delete un-sticks on the load AFTER the
       one that persisted them. Read unconditionally rather than behind
       `hydrated`: an unhydrated record still gets reconciled, which is exactly
       when the push-back fires. */
    deletedRunIds: Array.isArray(state.deletedRunIds)
      ? state.deletedRunIds.filter((id): id is string => typeof id === "string" && Boolean(id))
      : [],
    /* Survives reload mid-run. If the run finished while the tab was away,
       the queued text returns to the DRAFT on load rather than auto-sending —
       a reload cannot witness the completion, and hydrated state can disguise
       a failed (or still-active) run as a clean one. */
    queuedFollowup: typeof state.queuedFollowup === "string" ? state.queuedFollowup : "",
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
    // Both hydrated. Never let an empty hydrated record (e.g. a degraded server
    // reconstruction) clobber a populated local transcript — an empty transcript
    // replacing a real one is never correct. Otherwise prefer the newer record.
    if (candidate.messages.length === 0 && current.messages.length > 0) {
      return current;
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
        status: message.status,
        errorReason: message.errorReason,
        runId: message.runId,
        steering: message.steering,
        steerId: message.steerId,
        durationSeconds: message.durationSeconds,
        // Reasoning deltas are live-stream scaffolding (each one supersedes the
        // previous); persisting them ballooned every snapshot POST during a run.
        // Filter them from the serialized copies only — in-memory state is untouched.
        progressEvents: (message.progressEvents ?? []).filter(
          (event) => event.event !== "trace.reasoning.delta"
        ),
        runEvents: (message.runEvents ?? []).filter(
          (event) => String(event.event_type || "").trim() !== "trace.reasoning.delta"
        ),
        // The reasoning-delta events are stripped above (live scaffolding), so persist the
        // ACCUMULATED thinking text as its own durable field — this is what the post-turn
        // "Thinking" expansion reads back after reload.
        reasoning:
          (message.reasoning ?? reasoningTextFromRunEvents(message.runEvents ?? [])) || undefined,
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
      deletedRunIds: conversation.deletedRunIds ?? [],
      queuedFollowup: conversation.queuedFollowup ?? "",
    },
  };
};

// Snapshot fingerprints are memoized per conversation-object identity: every state
// mutation in this codebase replaces the ConversationState object, so an unchanged
// identity means an unchanged record. This keeps the debounced snapshot flush at
// O(1) per unchanged conversation instead of re-serializing every hydrated
// transcript (all runEvents/progressEvents) at ~1-2Hz during streaming.
const conversationRecordFingerprints = new WeakMap<
  ConversationState,
  { record: ConversationRecord; fingerprint: string }
>();

const recordAndFingerprintFor = (
  conversation: ConversationState
): { record: ConversationRecord; fingerprint: string } => {
  const cached = conversationRecordFingerprints.get(conversation);
  if (cached) {
    return cached;
  }
  const record = conversationToRecord(conversation);
  const entry = { record, fingerprint: JSON.stringify(record) };
  conversationRecordFingerprints.set(conversation, entry);
  return entry;
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

type ConversationTranscriptActions = {
  onStopConversation: () => void;
  onStreamingRenderComplete: (messageId: string) => void;
  onCopy: (value: string, feedbackKey?: string) => Promise<void>;
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
  /* Takes the id as well as the text: editing now truncates the turn and
     re-runs from that point, so it has to know WHICH message it is editing —
     the old signature only carried content, which is why "Edit" could do
     nothing but overwrite the composer draft. */
  onEditUserMessage: (messageId: string, content: string) => void;
  /* "Request" rather than "do": destructive actions route through a
     confirmation that names the blast radius, since deleting a user message
     also removes the reply it produced. */
  onRequestDeleteUserMessage: (messageId: string) => void;
  onRetryAssistant: (assistantMessageId: string, options: { edit: boolean }) => void;
  /* Toggles the report canvas for a report document's path key. Lives in
     actions (stable identity) so opening a canvas does not rebuild the
     transcript's callback plumbing. */
  onOpenReportDocument: (document: RunDocumentArtifact) => void;
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
  /* Which report (by conversation-level path key) is open in the canvas, and
     how many registrations each report path has across the conversation. Both
     are conversation-level facts a single message cannot know. */
  openReportPathKey: string | null;
  reportVersionCounts: Record<string, number>;
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
    openReportPathKey,
    reportVersionCounts,
  }: ConversationMessageRowProps) {
    const isAssistant = message.role === "assistant";
    const isCopied = copiedMessageId === message.id;
    const progressEvents = message.progressEvents ?? EMPTY_PROGRESS_EVENTS;
    const runEvents = message.runEvents ?? EMPTY_RUN_EVENTS;
    const runArtifacts = message.runArtifacts ?? EMPTY_RUN_IMAGE_ARTIFACTS;
    const runDocuments = message.runDocuments ?? EMPTY_RUN_DOCUMENTS;
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
          uploadPreviewUrlForFile
        ),
      [
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
    const hasPrimaryToolCard = toolResultCards.length > 0;
    const showLeadingToolResultCards =
      leadingToolResultCards.length > 0 && (!isStreamingAssistant || !message.liveStream);
    const showTrailingToolResultCards =
      trailingToolResultCards.length > 0 && (!isStreamingAssistant || !message.liveStream);
    const thinkingBarText = useMemo(
      () => thinkingBarTextForRunEvents(runEvents, isStreamingAssistant),
      [isStreamingAssistant, runEvents]
    );
    const elapsedLabel = useMemo(
      () => formatElapsedDuration(message.durationSeconds),
      [message.durationSeconds]
    );
    const tokenUsage = useMemo(() => extractRunTokenUsage(message), [message]);
    // A multi-step agentic run (it ran tools / executed code) shows the calm step timeline as the
    // primary surface and folds its running first-person narration into an opt-in "live reasoning"
    // disclosure — so a scientist watching a long run sees structured progress, not a monologue.
    // A plain text reply keeps streaming inline (responsive). Monotonic, so no flip-flop mid-run.
    const isAgenticRun = useMemo(() => runHasToolActivity(runEvents), [runEvents]);
    // The turn's chain-of-thought for the post-completion "Thought process" disclosure: message.reasoning
    // once persisted/rehydrated, else derived from the (still-in-memory) coalesced reasoning run event.
    const persistedReasoning = useMemo(
      () => (message.reasoning ?? "").trim() || reasoningTextFromRunEvents(runEvents),
      [message.reasoning, runEvents]
    );
    const showAssistantMetadataLine = Boolean(elapsedLabel) || Boolean(tokenUsage);
    if (!isAssistant) {
      return (
        <Message
          /* Row identity for ⌘F: highlight painting and scroll-into-view find
             mounted rows by this attribute, whichever transcript mode is live. */
          data-message-id={message.id}
          className={cn(
            "chat-width-frame mx-auto w-full justify-end px-4 sm:px-6",
            isLastMessage && "pk-message-enter"
          )}
        >
          <div className="group flex w-full flex-col items-end gap-1">
            {message.steering ? (
              /* Steering lifecycle, stated quietly above the bubble. aria-live
                 lets the pending→applied transition announce itself. */
              <span
                className="chat-steering-eyebrow"
                data-steering={message.steering}
                aria-live="polite"
              >
                {message.steering === "pending"
                  ? "Steering — will be seen shortly"
                  : message.steering === "applied"
                    ? "Seen by the agent"
                    : message.steering === "missed"
                      ? "Run ended before this was read — carries into the next turn"
                      : "Steered mid-run"}
              </span>
            ) : null}
            <MessageContent className="max-w-full bg-muted text-primary rounded-3xl px-5 py-2.5">
              {message.content}
            </MessageContent>
            {message.uploadedFileNames?.length ? (
              <p className="panel-caption">
                Attached: {message.uploadedFileNames.join(", ")}
              </p>
            ) : null}
            {/* data-pinned keeps the last turn's actions visible without a hover:
                it is the one row a user reaches for repeatedly. Everything else
                about the reveal — touch, keyboard, motion — is owned by
                .chat-message-actions in styles.css. */}
            <MessageActions
              className="chat-message-actions"
              data-role="user"
              data-pinned={isLastMessage || undefined}
            >
              <MessageAction tooltip="Edit">
                <button
                  type="button"
                  className="chat-message-action"
                  aria-label="Edit message"
                  onClick={() => actions.onEditUserMessage(message.id, message.content)}
                >
                  <Pencil className="size-4" aria-hidden="true" />
                </button>
              </MessageAction>
              <MessageAction tooltip="Delete">
                <button
                  type="button"
                  className="chat-message-action"
                  data-tone="destructive"
                  aria-label="Delete message"
                  onClick={() => actions.onRequestDeleteUserMessage(message.id)}
                >
                  <Trash className="size-4" aria-hidden="true" />
                </button>
              </MessageAction>
              {/* The accessible name carries the confirmation too, so the state
                  change is announced and not only drawn. */}
              <MessageAction tooltip={isCopied ? "Copied!" : "Copy"}>
                <button
                  type="button"
                  className="chat-message-action"
                  data-state={isCopied ? "copied" : undefined}
                  aria-label={isCopied ? "Message copied" : "Copy message"}
                  onClick={() => void actions.onCopy(message.content, message.id)}
                >
                  {isCopied ? (
                    <Check
                      className="size-4 animate-in zoom-in-50 fade-in-0 duration-200"
                      aria-hidden="true"
                    />
                  ) : (
                    <Copy className="size-4" aria-hidden="true" />
                  )}
                </button>
              </MessageAction>
            </MessageActions>
          </div>
        </Message>
      );
    }

    return (
      <Message
        data-message-id={message.id}
        className={cn(
          "chat-width-frame mx-auto w-full justify-start px-4 sm:px-6",
          isLastMessage && "pk-message-enter"
        )}
      >
        <div className="group flex w-full flex-1 flex-col gap-2">
          {showAssistantMetadataLine ? (
            <div className="text-muted-foreground flex flex-wrap items-center gap-2 text-xs leading-5">
              {tokenUsage || elapsedLabel ? (
                <span
                  className="tabular-nums"
                  title={
                    tokenUsage
                      ? `${tokenUsage.input_tokens.toLocaleString()} input · ${tokenUsage.output_tokens.toLocaleString()} output${
                          tokenUsage.model ? ` · ${tokenUsage.model}` : ""
                        }`
                      : undefined
                  }
                >
                  {tokenUsage ? (
                    isStreamingAssistant ? (
                      <AnimatedTokenCount value={tokenUsage.total_tokens} />
                    ) : (
                      `${formatTokens(tokenUsage.total_tokens)} tokens`
                    )
                  ) : null}
                  {tokenUsage && elapsedLabel ? " · " : null}
                  {elapsedLabel}
                </span>
              ) : null}
            </div>
          ) : null}
          {isStreamingAssistant ? (
            <div className="mb-1">
              <Suspense
                fallback={
                  <ThinkingBar
                    text={thinkingBarText ?? undefined}
                    onStop={actions.onStopConversation}
                    stopLabel="Stop"
                  />
                }
              >
                <LazyChatRunSteps
                  runEvents={runEvents}
                  progressEvents={progressEvents}
                  isStreaming={isStreamingAssistant}
                  statusText={thinkingBarText}
                  onStop={actions.onStopConversation}
                  stopLabel="Stop"
                />
              </Suspense>
            </div>
          ) : persistedReasoning ? (
            // Turn complete: the live step timeline is gone, so offer the persisted reasoning as a
            // collapsed-by-default disclosure the reader can open.
            <div className="mb-1">
              <ReasoningTrace text={persistedReasoning} />
            </div>
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
            <LiveStreamRegion
              messageId={message.id}
              liveStream={message.liveStream}
              foldIntoReasoning={isAgenticRun}
              onComplete={() => actions.onStreamingRenderComplete(message.id)}
            />
          ) : (
            <MessageContent
              className="w-full bg-transparent p-0 text-foreground"
              id={message.id}
              markdown
            >
              {displayContent}
            </MessageContent>
          )}
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
          {runArtifacts.length > 0 && toolResultCards.length === 0 ? (
            <div className="chat-artifact-grid">
              {runArtifacts.map((artifact) => (
                <div key={artifact.path} className="chat-artifact-cell">
                  <button
                    type="button"
                    className="chat-artifact-card"
                    onClick={() => {
                      const figures = runArtifactsToFigures(runArtifacts);
                      openFigureLightbox(
                        figures,
                        figures.findIndex((figure) => figure.url === artifact.url)
                      );
                    }}
                    disabled={!artifact.previewable}
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
                  </button>
                  {artifact.previewable && message.runId ? (
                    <FigureCaption runId={message.runId} path={artifact.path} apiClient={apiClient} />
                  ) : null}
                </div>
              ))}
            </div>
          ) : null}
          {runDocuments.length > 0 ? (
            <Suspense fallback={null}>
              <LazyChatRunDocuments
                documents={runDocuments}
                openReportPathKey={openReportPathKey}
                reportVersionCounts={reportVersionCounts}
                onOpenReport={actions.onOpenReportDocument}
              />
            </Suspense>
          ) : null}
          {message.status === "stopped" || message.status === "failed" ? (
            <AssistantTurnRecovery
              status={message.status}
              errorReason={message.errorReason}
              onRetry={() => actions.onRetryAssistant(message.id, { edit: false })}
              onEdit={() => actions.onRetryAssistant(message.id, { edit: true })}
            />
          ) : null}
          {/* Upvote/Downvote used to sit here. They were pure decoration — no
              onClick, no state, no endpoint behind them at any layer. Once the
              row became visible on touch (see .chat-message-actions) they would
              have been two permanently-visible controls that do nothing on every
              phone, and a control that hovers and tooltips but never responds
              costs more trust than an absent one. They come back with the
              feedback slice: keyed on run_id, not message.id, because the
              live-turn id here is a client UUID the server replaces on persist. */}
          <MessageActions
            className="chat-message-actions"
            data-role="assistant"
            data-pinned={isLastMessage || undefined}
          >
            <MessageAction tooltip={isCopied ? "Copied!" : "Copy"}>
              <button
                type="button"
                className="chat-message-action"
                data-state={isCopied ? "copied" : undefined}
                aria-label={isCopied ? "Response copied" : "Copy response"}
                disabled={isStreamingAssistant}
                onClick={() => void actions.onCopy(message.content, message.id)}
              >
                {isCopied ? (
                  <Check
                    className="size-4 animate-in zoom-in-50 fade-in-0 duration-200"
                    aria-hidden="true"
                  />
                ) : (
                  <Copy className="size-4" aria-hidden="true" />
                )}
              </button>
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
    previousProps.apiClient === nextProps.apiClient &&
    /* Same rule as the transcript comparator above: rendered-from props must
       be listed, or the report card's open/version chrome freezes at first
       paint (both narrow comparators shipped stale in live verification —
       fiber inspection showed the memo boundary holding the new key while
       the row inside rendered the old null). */
    previousProps.openReportPathKey === nextProps.openReportPathKey &&
    previousProps.reportVersionCounts === nextProps.reportVersionCounts
);

// Rotating, composer-forward invitations shown on the empty/new-chat screen. Kept
// in our calm voice: sentence case, warm, no exclamation. Advanced sequentially by a
// nonce that bumps on every "New chat" action, so each new chat shows the next one
// (no consecutive repeats). Nonce starts at 0 on load, so the first thing a fresh
// session sees is WELCOME_PROMPTS[0].
const WELCOME_PROMPTS = [
  "What are we working on?",
  "Where should we start?",
  "What are you exploring today?",
  "What can I help you investigate?",
  "What should we look at first?",
  "What's the question?",
] as const;

function welcomePromptForNonce(nonce: number): string {
  const len = WELCOME_PROMPTS.length;
  return WELCOME_PROMPTS[((nonce % len) + len) % len];
}

// Best-effort first name from an identity string (usually an email). "amil@ucsb.edu"
// -> "Amil". Returns null when nothing usable can be derived so the greeting is skipped.
function deriveFirstName(identity: string | null): string | null {
  if (!identity) return null;
  const local = identity.split("@")[0]?.trim() ?? "";
  const first = local.split(/[.\-_+]/)[0]?.trim() ?? "";
  if (!first || /\d/.test(first)) return null;
  return first.charAt(0).toUpperCase() + first.slice(1).toLowerCase();
}

type ConversationTranscriptProps = {
  conversationHydrated: boolean;
  isPhoneView: boolean;
  welcomeName: string | null;
  welcomeNonce: number;
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
  openReportPathKey: string | null;
  reportVersionCounts: Record<string, number>;
  /* ⌘F navigation: which message to bring into view. The nonce forces a
     re-scroll when the user re-navigates to the same match after scrolling
     away — identity alone would look unchanged. */
  findTarget: { messageId: string; messageIndex: number; nonce: number } | null;
};

const ConversationTranscript = memo(
  function ConversationTranscript({
    conversationHydrated,
    isPhoneView,
    welcomeName,
    welcomeNonce,
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
    openReportPathKey,
    reportVersionCounts,
    findTarget,
  }: ConversationTranscriptProps) {
    const conversationRunArtifacts = useMemo(
      () => collectConversationRunArtifacts(messages),
      [messages]
    );
    const { scrollRef } = useStickToBottomContext();
    const [virtualizedScrollParent, setVirtualizedScrollParent] = useState<HTMLElement | null>(null);
    useLayoutEffect(() => {
      setVirtualizedScrollParent(scrollRef.current);
    }, [scrollRef]);

    // Short chats keep the normal stick-to-bottom content path. Long hydrated transcripts switch to
    // react-virtuoso so all messages remain reachable while the mounted DOM stays bounded.
    const shouldVirtualizeMessages = messages.length > MESSAGE_WINDOW_SIZE;
    const [messageWindow, setMessageWindow] = useState(MESSAGE_WINDOW_SIZE);
    const firstMessageId = messages.length > 0 ? messages[0].id : null;
    const prevFirstMessageIdRef = useRef(firstMessageId);
    useEffect(() => {
      if (prevFirstMessageIdRef.current !== firstMessageId) {
        prevFirstMessageIdRef.current = firstMessageId;
        setMessageWindow(MESSAGE_WINDOW_SIZE);
      }
    }, [firstMessageId]);
    const { visible: visibleMessages, hiddenCount: hiddenMessageCount } = windowTailMessages(
      messages,
      messageWindow
    );
    const virtuosoRef = useRef<VirtuosoHandle | null>(null);
    /* ⌘F: bring the current match's message into view. Both transcript modes
       hide messages from the DOM — Virtuoso virtualizes long chats, and the
       windowed path hides the head behind "Show earlier messages" — so each
       needs its own route there. Highlighting is NOT done here: App retries
       paint until this scroll has actually mounted the row. */
    useEffect(() => {
      if (!findTarget) {
        return;
      }
      const index =
        messages[findTarget.messageIndex]?.id === findTarget.messageId
          ? findTarget.messageIndex
          : messages.findIndex((message) => message.id === findTarget.messageId);
      if (index < 0) {
        return;
      }
      if (shouldVirtualizeMessages) {
        virtuosoRef.current?.scrollToIndex({ index, align: "center" });
        return;
      }
      const hiddenCount = Math.max(messages.length - messageWindow, 0);
      if (index < hiddenCount) {
        // The match is behind the "Show earlier messages" fold — widen the
        // window exactly far enough to include it.
        setMessageWindow(messages.length - index);
      }
      window.requestAnimationFrame(() => {
        document
          .querySelector(`[data-message-id="${CSS.escape(findTarget.messageId)}"]`)
          ?.scrollIntoView({ block: "center" });
      });
      // Deliberately narrow: re-running on every messages identity change would
      // yank the scroll position on each streaming delta while find is open.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [findTarget]);
    const renderMessageRow = useCallback(
      (message: UiMessage, index: number, totalMessages: number) => (
        <ErrorBoundary
          key={message.id}
          source="message-row"
          resetKeys={[message.id, message.content, message.runId]}
        >
          <ConversationMessageRow
            message={message}
            isLastMessage={index === totalMessages - 1}
            isStreamingAssistant={streamingMessageId === message.id}
            copiedMessageId={copiedMessageId}
            conversationRunArtifacts={conversationRunArtifacts}
            uploadedFiles={uploadedFiles}
            bisqueLinksByFileId={bisqueLinksByFileId}
            apiClient={apiClient}
            actions={actions}
            openReportPathKey={openReportPathKey}
            reportVersionCounts={reportVersionCounts}
          />
        </ErrorBoundary>
      ),
      [
        actions,
        apiClient,
        bisqueLinksByFileId,
        conversationRunArtifacts,
        copiedMessageId,
        openReportPathKey,
        reportVersionCounts,
        streamingMessageId,
        uploadedFiles,
      ]
    );
    const welcomePrompt = welcomePromptForNonce(welcomeNonce);
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
            {isPhoneView ? (
              <div className="mobile-chat-hero">
                <h2 className="mobile-chat-hero-title">What can I help with?</h2>
                <p className="mobile-chat-hero-subtitle">
                  Ask anything below — add an image or a resource whenever you like.
                </p>
                <details className="mobile-usage-disclosure">
                  <summary className="mobile-usage-summary">
                    <span>Your usage</span>
                    <ChevronDown className="mobile-usage-chevron size-4" aria-hidden />
                  </summary>
                  <div className="mobile-usage-body">
                    <UserTokenUsagePanel
                      tokenUsage={blankChatTokenUsage}
                      loading={blankChatUsageLoading}
                      error={blankChatUsageError}
                      className="blank-chat-usage-panel"
                      density="compact"
                    />
                  </div>
                </details>
              </div>
            ) : (
              <div className="blank-chat-welcome">
                <div className="blank-chat-welcome-greeting">
                  {welcomeName ? (
                    <p className="blank-chat-welcome-eyebrow">
                      Welcome back, {welcomeName}
                    </p>
                  ) : null}
                  <h2 className="blank-chat-welcome-hero">{welcomePrompt}</h2>
                </div>
                <details className="blank-chat-usage-disclosure">
                  <summary className="blank-chat-usage-strip">
                    {blankChatTokenUsage?.summary ? (
                      <span className="blank-chat-usage-strip-stats">
                        <span>
                          <strong>
                            {formatTokens(
                              blankChatTokenUsage.summary.lifetime_total_tokens
                            )}
                          </strong>{" "}
                          tokens
                        </span>
                        <span className="blank-chat-usage-strip-dot" aria-hidden>
                          ·
                        </span>
                        <span>
                          <strong>
                            {blankChatTokenUsage.summary.current_streak_days}-day
                          </strong>{" "}
                          streak
                        </span>
                        <span className="blank-chat-usage-strip-dot" aria-hidden>
                          ·
                        </span>
                      </span>
                    ) : null}
                    <span className="blank-chat-usage-strip-toggle">
                      View usage
                      <ChevronDown
                        className="blank-chat-usage-strip-chevron size-4"
                        aria-hidden
                      />
                    </span>
                  </summary>
                  <div className="blank-chat-usage-disclosure-body">
                    <UserTokenUsagePanel
                      tokenUsage={blankChatTokenUsage}
                      loading={blankChatUsageLoading}
                      error={blankChatUsageError}
                      className="blank-chat-usage-panel"
                      density="compact"
                    />
                  </div>
                </details>
              </div>
            )}
          </div>
        ) : shouldVirtualizeMessages && virtualizedScrollParent ? (
          // Only mount Virtuoso once the StickToBottom scroll element is captured — otherwise
          // customScrollParent is undefined and Virtuoso would spin up its OWN nested scroller
          // inside the existing scroll area (a broken double-scrollbar) for the first paint. Until
          // then the bounded-window fallback below renders (already DOM-bounded), then it swaps in.
          <Suspense fallback={null}>
            <LazyVirtuoso
              ref={virtuosoRef}
              className="w-full"
              customScrollParent={virtualizedScrollParent}
              data={messages}
              computeItemKey={(_: number, message: UiMessage) => message.id}
              followOutput="auto"
              initialTopMostItemIndex={messages.length - 1}
              increaseViewportBy={{ top: 900, bottom: 1_200 }}
              itemContent={(index: number, message: UiMessage) =>
                renderMessageRow(message, index, messages.length)
              }
            />
          </Suspense>
        ) : (
          <>
            {hiddenMessageCount > 0 ? (
              <button
                type="button"
                onClick={() => setMessageWindow((size) => size + MESSAGE_WINDOW_SIZE)}
                className="mx-auto mb-2 flex items-center justify-center rounded-full border border-[var(--line)] bg-[var(--bg-panel)] px-4 py-1.5 text-[12px] font-medium text-[var(--text-muted)] transition-colors hover:text-[var(--text-main)]"
              >
                {`Show earlier messages (${hiddenMessageCount.toLocaleString()})`}
              </button>
            ) : null}
            {visibleMessages.map((message, index) => (
              renderMessageRow(message, index, visibleMessages.length)
            ))}
          </>
        )}
      </ChatContainerContent>
    );
  },
  (previousProps, nextProps) =>
    previousProps.conversationHydrated === nextProps.conversationHydrated &&
    previousProps.isPhoneView === nextProps.isPhoneView &&
    previousProps.welcomeName === nextProps.welcomeName &&
    previousProps.welcomeNonce === nextProps.welcomeNonce &&
    previousProps.messages === nextProps.messages &&
    previousProps.findTarget === nextProps.findTarget &&
    previousProps.blankChatTokenUsage === nextProps.blankChatTokenUsage &&
    previousProps.blankChatUsageLoading === nextProps.blankChatUsageLoading &&
    previousProps.blankChatUsageError === nextProps.blankChatUsageError &&
    previousProps.streamingMessageId === nextProps.streamingMessageId &&
    previousProps.copiedMessageId === nextProps.copiedMessageId &&
    previousProps.uploadedFiles === nextProps.uploadedFiles &&
    previousProps.bisqueLinksByFileId === nextProps.bisqueLinksByFileId &&
    previousProps.apiClient === nextProps.apiClient &&
    /* The report canvas' open state must reach the cards: this comparator is
       deliberately narrow (actions identity churn is tolerated as stale), so
       any prop the rows RENDER from has to be listed here explicitly — the
       card's open/version chrome went permanently stale without these. The
       re-render also refreshes the rows' actions closure, which is what lets
       a second click on an open card read the CURRENT canvas target and
       toggle it closed. */
    previousProps.openReportPathKey === nextProps.openReportPathKey &&
    previousProps.reportVersionCounts === nextProps.reportVersionCounts
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

// Optimistic, placeholder-only title shown while the model-generated title is
// still being produced (and as a calm last resort if generation fails). A
// readable trim of the prompt's first sentence reads far better than the old
// keyword-mangled summary ("Describe Image Inspect Tell ...").
const summarizeConversationTitle = (value: string): string => {
  const singleLine = value
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^["'`]+|["'`]+$/g, "");
  if (!singleLine) {
    return "New conversation";
  }
  const firstSentence = singleLine.split(/(?<=[.?!])\s+/)[0] || singleLine;
  const base = firstSentence.length >= 12 ? firstSentence : singleLine;
  if (base.length <= 48) {
    return base;
  }
  const truncated = base.slice(0, 47);
  const atWord = truncated.includes(" ")
    ? truncated.slice(0, truncated.lastIndexOf(" "))
    : truncated;
  return `${atWord.trim()}…`;
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

// The agent emits the REAL backend tool name (e.g. bisque_search_resources); a few
// result-card branches below were written against the slash-menu name instead. Map
// real → card name so those results render. (bisque_download_resource and
// bisque_create_dataset already match by their real names, so they're not aliased.)
const TOOL_NAME_CARD_ALIASES: Record<string, string> = {
  bisque_search_resources: "search_bisque_resources",
  bisque_upload_files: "upload_to_bisque",
  bisque_upload_workspace_files: "upload_to_bisque",
};

const normalizeToolName = (value: unknown): string => {
  if (typeof value !== "string") {
    return "";
  }
  const trimmed = value.trim();
  return TOOL_NAME_CARD_ALIASES[trimmed] ?? trimmed;
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

const streamTokenDeliveryKey = (
  conversationId: string,
  messageId: string,
  event?: StreamTokenEvent
): string | null => {
  const sequence = Math.floor(Number(event?.sequence ?? 0));
  if (!Number.isFinite(sequence) || sequence <= 0) {
    return null;
  }
  const runId = String(event?.runId ?? "").trim();
  return `${conversationId}:${messageId}:${runId}:${sequence}`;
};

const shouldApplyStreamToken = (
  delivered: Map<string, true>,
  conversationId: string,
  messageId: string,
  event?: StreamTokenEvent
): boolean => {
  const key = streamTokenDeliveryKey(conversationId, messageId, event);
  if (!key) {
    return true;
  }
  if (delivered.has(key)) {
    return false;
  }
  delivered.set(key, true);
  return true;
};

const clearStreamTokenDeliveries = (
  delivered: Map<string, true>,
  conversationId: string,
  messageId: string
): void => {
  const prefix = `${conversationId}:${messageId}:`;
  Array.from(delivered.keys()).forEach((key) => {
    if (key.startsWith(prefix)) {
      delivered.delete(key);
    }
  });
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
    `/v2/uploads/${encodeURIComponent(fileId)}/preview`
): ToolResultCard[] => {
  if (!progressEvents.length) {
    return [];
  }
  type BisqueSearchCandidate = {
    index: number;
    toolName: "search_bisque_resources";
    matchCount: number | null;
    resourceType?: string;
    resourceRows: ToolResourceRow[];
  };

  const artifactBySource = new Map<string, RunImageArtifact[]>();
  const uploadedPreviewLookup = buildUploadedArtifactPreviewLookup(uploadedFiles);
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
  const bisqueDownloadRows: ToolDownloadRow[] = [];
  let latestBisqueDownloadIndex = -1;
  let latestBisqueUploadCard: ToolResultCard | null = null;
  progressEvents.forEach((event, index) => {
    if (event.event !== "completed") {
      return;
    }
    const toolName = normalizeToolName(event.tool);
    if (
      toolName !== "yolo_detect" &&
      toolName !== "upload_to_bisque" &&
      toolName !== "bisque_download_resource" &&
      toolName !== "bisque_create_dataset" &&
      toolName !== "search_bisque_resources"
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

    if (toolName === "search_bisque_resources") {
      if (summary?.success === false) {
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
      const matchCount = toNumber(summary?.count);
      const resourceType =
        typeof summary?.resource_type === "string" && summary.resource_type.trim()
          ? summary.resource_type.trim()
          : undefined;

      const resourceTypeKey = (resourceType || "resource").toLowerCase();
      const candidate: BisqueSearchCandidate = {
        index,
        toolName: "search_bisque_resources",
        matchCount,
        resourceType,
        resourceRows,
      };
      const existing = bisqueSearchByType.get(resourceTypeKey);
      if (!existing) {
        bisqueSearchByType.set(resourceTypeKey, candidate);
        return;
      }
      const mergedRows = dedupeBisqueResourceRows(
        [...existing.resourceRows, ...candidate.resourceRows],
        12
      );
      const mergedCount = Math.max(
        mergedRows.length,
        existing.matchCount ?? 0,
        candidate.matchCount ?? 0
      );
      bisqueSearchByType.set(resourceTypeKey, {
        index: Math.max(existing.index, candidate.index),
        toolName: "search_bisque_resources",
        matchCount: mergedCount,
        resourceType: candidate.resourceType ?? existing.resourceType,
        resourceRows: mergedRows,
      });
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

    if (toolName === "bisque_create_dataset") {
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
          : "created";
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

  if (bisqueSearchByType.size > 0) {
    const selectedBisqueSearches = Array.from(bisqueSearchByType.values()).sort(
      (left, right) => left.index - right.index
    );
    const mergedRows = dedupeBisqueResourceRows(
      selectedBisqueSearches.flatMap((candidate) => candidate.resourceRows),
      12
    );
    const totalMatches = selectedBisqueSearches.reduce((sum, candidate) => {
      const countValue = candidate.matchCount ?? candidate.resourceRows.length;
      return sum + Math.max(0, Math.round(countValue));
    }, 0);
    if (totalMatches <= 0 && mergedRows.length === 0) {
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
      mergedBisqueSearchCard = {
        id: `bisque-search-${lastCandidate.index}`,
        tool: "search_bisque_resources",
        title: "BisQue search",
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

  if (mergedBisqueSearchCard && !primaryBisqueCard) {
    primaryBisqueCard = mergedBisqueSearchCard;
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

const extractSearchResourceRowsFromMessage = (message: UiMessage): ToolResourceRow[] => {
  const cards = buildToolResultCards(message.progressEvents ?? [], message.runArtifacts ?? []);
  return dedupeBisqueResourceRows(
    cards
      .filter((card) => card.tool === "search_bisque_resources")
      .flatMap((card) => card.resourceRows)
      .filter((row) => Boolean(row.resourceUri))
  );
};

const extractResolvedBisqueRowsFromMessage = (message: UiMessage): ToolResourceRow[] =>
  dedupeBisqueResourceRows(message.resolvedBisqueResources ?? []);

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
  if (
    /\bdataset\b/.test(lowered) &&
    /\b(create|make|build|assemble|call(?:ed)?|named?)\b/.test(lowered)
  ) {
    selected.add("bisque_create_dataset");
  }
  if (
    /\bdataset\b/.test(lowered) &&
    /\b(add|append|put|organize|move|save)\b/.test(lowered) &&
    (options?.hasStagedUploads || isBisqueUploadActionPrompt(promptText, options))
  ) {
    selected.add("upload_to_bisque");
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
  // No BisQue deletion tool exists, so a turn can never clear the selection.
  const clearsSelection = false;
  const resolvedRows = dedupeBisqueResourceRows(
    toolResultCards
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
  // The per-phase copy branches this used to key on (memory/knowledge/learning/
  // tool_event/pro_mode/graph_event) matched event kinds no deployed backend has
  // ever emitted; ChatRunSteps owns the real per-step copy.
  void runEvents;
  return isStreaming ? DEFAULT_THINKING_TEXT : null;
};

// Compact elapsed-time label for the assistant metadata line, e.g. "8s",
// "1m 12s", "2h 3m". Returns null when there is no positive duration.
const formatElapsedDuration = (seconds: number | null | undefined): string | null => {
  const value = Number(seconds ?? 0);
  if (!Number.isFinite(value) || value <= 0) {
    return null;
  }
  if (value < 60) {
    return `${Math.max(1, Math.round(value))}s`;
  }
  if (value < 3600) {
    const minutes = Math.floor(value / 60);
    const secs = Math.round(value % 60);
    return secs > 0 ? `${minutes}m ${secs}s` : `${minutes}m`;
  }
  const hours = Math.floor(value / 3600);
  const minutes = Math.round((value % 3600) / 60);
  return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
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
  };
};

function ComposerAttachMenu({
  disabled,
  variant = "toolbar",
  onCloseAutoFocus,
}: {
  disabled: boolean;
  variant?: "toolbar" | "idle";
  onCloseAutoFocus?: (event: Event) => void;
}) {
  const { openFilePicker, openFolderPicker, allowDirectories } = useFileUploadContext();
  // The idle variant is the slim pill's left affordance. It stays a real tab
  // stop (attach is a primary action and, while slim, the toolbar's + is
  // visibility:hidden — so this is the only keyboard path to it); its slim CSS
  // gates it out of the tab order + a11y tree when the toolbar + takes over.
  // mousedown preventDefault only stops a MOUSE click from blurring the caret.
  // Both variants carry the tooltip: a bare + is the least self-evident control
  // on the bar, and the tip is where "this also takes folders" is discoverable
  // without opening the menu.
  const trigger = (
    <DropdownMenuTrigger asChild>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        aria-label="Attach files or a folder"
        data-testid={variant === "idle" ? "composer-idle-attach-menu" : "composer-attach-menu"}
        onMouseDown={variant === "idle" ? (event) => event.preventDefault() : undefined}
        className={
          variant === "idle"
            ? "app-composer-idle-attach app-composer-icon-button size-10 rounded-full"
            : "app-composer-icon-button composer-attach-button size-11 rounded-full sm:size-10"
        }
        disabled={disabled}
      >
        <Plus size={18} />
      </Button>
    </DropdownMenuTrigger>
  );
  return (
    <DropdownMenu>
      <PromptInputAction
        tooltip="Attach files or a folder"
        disabled={disabled}
        side="top"
        sideOffset={8}
        delayDuration={350}
        className="app-composer-tooltip"
      >
        {trigger}
      </PromptInputAction>
      <DropdownMenuContent
        align="start"
        sideOffset={8}
        className="app-composer-attach-menu"
        onCloseAutoFocus={onCloseAutoFocus}
      >
        <DropdownMenuItem onSelect={() => openFilePicker()}>
          <FileUp data-icon="inline-start" aria-hidden="true" />
          <div className="app-composer-attach-menu-item">
            <span>Files</span>
            <span className="app-composer-attach-menu-detail">Images, tables, documents</span>
          </div>
        </DropdownMenuItem>
        {allowDirectories ? (
          <DropdownMenuItem onSelect={() => openFolderPicker()}>
            <FolderUp data-icon="inline-start" aria-hidden="true" />
            <div className="app-composer-attach-menu-item">
              <span>Folder</span>
              <span className="app-composer-attach-menu-detail">
                OME-Zarr uploads as one dataset
              </span>
            </div>
          </DropdownMenuItem>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function App() {
  const apiBaseUrl = DEFAULT_API_BASE_URL;
  const [themePreference, setThemePreference] = useLocalStorageState<ThemePreference>(
    "bisque.frontend.themePreference",
    "system"
  );
  const resolvedTheme = useThemePreference(themePreference);
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
  const [settingsInitialTab, setSettingsInitialTab] =
    useState<SettingsTab>("general");
  const openSettings = useCallback(
    (tab: SettingsTab = "general") => {
      void loadAppSettingsDialogModule();
      setSettingsInitialTab(tab);
      setSettingsDialogOpen(true);
    },
    []
  );
  const isPhoneView = useBreakpoint(641);
  // Phone-only: when the user scrolls up to read a long answer, collapse the
  // composer to reclaim reading space (expands on focus / at-bottom / sending).
  const [composerScrolledAway, setComposerScrolledAway] = useState(false);

  /* Report canvas — the reading surface for run-generated reports. Two
     regimes, decided by the MAIN SHELL's own width (an expanded sidebar can
     narrow the stage inside a wide window, so the viewport is the wrong
     axis): a true split whenever the stage affords the transcript and panel
     minimums together, a full-screen sheet everywhere else. There is
     deliberately no floating overlay between them — it half-covered the
     transcript and turned the sidebar into a competing overlay. */
  /* A STATE ref, not useRef + []-effect: the shell mounts after the auth
     gate resolves, and a mount-once effect would observe nothing forever. */
  const [mainShellElement, setMainShellElement] = useState<HTMLElement | null>(null);
  const mainShellWidthRef = useRef<number | null>(null);
  const [mainShellWidth, setMainShellWidth] = useState<number | null>(null);
  useEffect(() => {
    if (!mainShellElement || typeof ResizeObserver === "undefined") {
      return;
    }
    const update = () => {
      const width = mainShellElement.clientWidth;
      mainShellWidthRef.current = width;
      /* Quantized so a live window-resize re-renders a handful of times,
         not per pixel. */
      setMainShellWidth(Math.round(width / 16) * 16);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(mainShellElement);
    return () => observer.disconnect();
  }, [mainShellElement]);
  const reportCanvasMode: ReportCanvasMode =
    isPhoneView ||
    (mainShellWidth !== null && mainShellWidth < REPORT_CANVAS_SPLIT_MIN_STAGE)
      ? "sheet"
      : "split";
  /* The divider commits here; the width survives sessions. Bounds re-derive
     from the live stage width so a resize can never starve the transcript. */
  const [reportCanvasStoredWidth, setReportCanvasStoredWidth] = useState<number | null>(() => {
    if (typeof window === "undefined") {
      return null;
    }
    try {
      const raw = window.localStorage.getItem(REPORT_CANVAS_WIDTH_STORAGE_KEY);
      const parsed = raw === null ? Number.NaN : Number.parseInt(raw, 10);
      return Number.isFinite(parsed) ? parsed : null;
    } catch {
      return null;
    }
  });
  const reportCanvasSplitBounds = useMemo(
    () => ({
      min: REPORT_CANVAS_PANEL_MIN,
      max: Math.max(
        REPORT_CANVAS_PANEL_MIN,
        Math.min(
          REPORT_CANVAS_PANEL_MAX,
          (mainShellWidth ?? 1280) - REPORT_CANVAS_TRANSCRIPT_MIN
        )
      ),
    }),
    [mainShellWidth]
  );
  const reportCanvasSplitWidth = Math.min(
    reportCanvasSplitBounds.max,
    Math.max(
      reportCanvasSplitBounds.min,
      reportCanvasStoredWidth ?? REPORT_CANVAS_PANEL_DEFAULT
    )
  );
  const handleReportCanvasWidthCommit = useCallback((width: number) => {
    const rounded = Math.round(width);
    setReportCanvasStoredWidth(rounded);
    try {
      window.localStorage.setItem(REPORT_CANVAS_WIDTH_STORAGE_KEY, String(rounded));
    } catch {
      /* Private-mode storage failures only cost persistence. */
    }
  }, []);
  const handleReportCanvasWidthReset = useCallback(() => {
    setReportCanvasStoredWidth(null);
    try {
      window.localStorage.removeItem(REPORT_CANVAS_WIDTH_STORAGE_KEY);
    } catch {
      /* Same: reset still applies for this session. */
    }
  }, []);
  /* Controlled so opening the canvas can collapse the sidebar to its icon
     rail (paying the width back to the transcript) and closing can restore
     it. The SidebarTrigger keeps working — it drives this same state through
     the provider's onOpenChange. */
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [reportCanvasTarget, setReportCanvasTarget] = useState<{
    conversationId: string;
    pathKey: string;
  } | null>(null);
  const [reportCanvasClosing, setReportCanvasClosing] = useState(false);
  const reportCanvasCloseTimerRef = useRef<number | null>(null);
  /* Auto-open fires once per report path per conversation — a later version
     of the same report never re-opens the canvas over the reader. */
  const reportCanvasAutoOpenedKeysRef = useRef<Set<string>>(new Set());
  /* What the sidebar was before the canvas collapsed it. STATE, not a ref:
     the conversation-switch reset reads it during render (the sanctioned
     reset-from-props pattern), and refs must not be read there. The mirror
     ref below serves event handlers that need the CURRENT sidebar state
     without re-memoizing the whole action chain on every sidebar toggle. */
  const [sidebarOpenBeforeCanvas, setSidebarOpenBeforeCanvas] = useState<boolean | null>(
    null
  );
  const sidebarOpenRef = useRef(true);
  useEffect(() => {
    sidebarOpenRef.current = sidebarOpen;
  }, [sidebarOpen]);
  const reportCanvasAutoOpenRef = useRef<
    ((conversationId: string, pathKeys: string[]) => void) | null
  >(null);

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
  /* repliesRemoved is captured at request time so the confirmation can state the
     real blast radius ("this also removes the reply it produced") rather than
     leaving the user to discover it. */
  const [pendingMessageDeletion, setPendingMessageDeletion] = useState<{
    messageId: string;
    repliesRemoved: number;
  } | null>(null);
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
  const [resourceQuery, setResourceQuery] = useState("");
  // Bumped by the mobile nav bar's search action; ResourceBrowser watches it and
  // reveals + focuses its search field. The Resources header is not sticky on
  // mobile, so this is what keeps search reachable from anywhere in the list.
  const [resourceSearchFocusSignal, setResourceSearchFocusSignal] = useState(0);
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
  const [adminMetrics, setAdminMetrics] = useState<AdminMetricsResponse | null>(null);
  const [adminLoadingMetrics, setAdminLoadingMetrics] = useState(false);
  const [adminMetricsRangeDays, setAdminMetricsRangeDays] = useState(90);
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
  // Set by "Retry" on a stopped/failed turn: after the stale pair is removed
  // (which triggers a re-render), an effect re-sends this prompt through the
  // normal submit pipeline. A ref avoids an extra render + cascading setState.
  const pendingRetryRef = useRef<{ conversationId: string; prompt: string } | null>(null);
  const [composerDraftsByConversationId, setComposerDraftsByConversationId] = useState<
    Record<string, string>
  >(() => readComposerDraftsFromStorage());

  const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const activeChatScrollElementRef = useRef<HTMLElement | null>(null);
  const conversationScrollMemoryRef = useRef<Record<string, ConversationScrollMemory>>({});
  const conversationScrollWriteBlockRef = useRef<string | null>(null);
  const persistedConversationHashesRef = useRef<Record<string, string>>({});
  const optimisticConversationIdsRef = useRef<Set<string>>(new Set());
  const hydratingConversationIdsRef = useRef<Set<string>>(new Set());
  const activeChatAbortControllersRef = useRef<Map<string, AbortController>>(new Map());
  const runStreamRecoveryControllersRef = useRef<Map<string, AbortController>>(new Map());
  const streamTokenDeliveriesRef = useRef<Map<string, true>>(new Map());
  const [localActiveRunIds, setLocalActiveRunIds] = useState<Set<string>>(() => new Set());
  const stopRequestedConversationIdsRef = useRef<Set<string>>(new Set());
  const copyFeedbackTimeoutRef = useRef<number | null>(null);
  const runArtifactHydrationsRef = useRef<Set<string>>(new Set());
  // Memoized shouldHydrateRunArtifacts verdicts keyed on the exact object
  // identities the decision reads (all replaced-not-mutated), so never-hydratable
  // messages stop rebuilding their tool cards on every conversations tick.
  const runArtifactHydrationDecisionsRef = useRef(
    new Map<
      string,
      {
        progressEvents: unknown;
        runArtifacts: unknown;
        responseMetadata: unknown;
        content: unknown;
        uploadedFiles: unknown;
        decision: boolean;
      }
    >()
  );
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
          .map((conversation) => [
            conversation.id,
            recordAndFingerprintFor(conversation).fingerprint,
          ])
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
    const activeChatAbortControllers = activeChatAbortControllersRef.current;
    const runStreamRecoveryControllers = runStreamRecoveryControllersRef.current;
    const streamTokenDeliveries = streamTokenDeliveriesRef.current;
    const stopRequestedConversationIds = stopRequestedConversationIdsRef.current;
    return () => {
      activeChatAbortControllers.forEach((controller) => controller.abort());
      activeChatAbortControllers.clear();
      runStreamRecoveryControllers.forEach((controller) => controller.abort());
      runStreamRecoveryControllers.clear();
      streamTokenDeliveries.clear();
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
      } catch (error) {
        if (isCancelled) {
          return;
        }
        // A failed bootstrap must be visible: silently seeding a blank draft
        // presented a populated account as "No history yet" until a manual
        // reload (list hydration is per-row resilient now, so reaching this
        // means the thread list itself failed).
        setUiErrorBanner(`Failed to load conversations: ${normalizeApiError(error)}`);
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
    const urlConversation = resolvedConversationId
      ? conversations.find((conversation) => conversation.id === resolvedConversationId) ?? null
      : null;
    const targetConversationId = shouldExposeConversationInUrl(urlConversation)
      ? resolvedConversationId
      : null;
    const currentUrlConversationId = readConversationIdFromLocation();
    if (targetConversationId === currentUrlConversationId) {
      return; // in sync (covers Back/Forward restores — never write a new entry for those)
    }
    if (targetConversationId && currentUrlConversationId) {
      // A real thread-to-thread switch: push so Back returns to the previous one.
      pushConversationIdInLocation(targetConversationId);
      return;
    }
    // First exposure or clearing a draft: normalize in place.
    replaceConversationIdInLocation(targetConversationId);
  }, [activeConversationId, authStatus, conversations, conversationsHydrated]);

  const flushConversationSnapshots = useCallback(() => {
    if (authStatus !== "authenticated" || !conversationsHydrated) {
      return;
    }
    const entries = conversations
      .filter(shouldPersistConversationSnapshot)
      .map(recordAndFingerprintFor);
    const previousHashes = persistedConversationHashesRef.current;
    const nextHashes: Record<string, string> = {};
    const changedRecords = entries
      .filter(({ record, fingerprint }) => {
        nextHashes[record.conversation_id] = fingerprint;
        return previousHashes[record.conversation_id] !== fingerprint;
      })
      .map(({ record }) => record);
    persistedConversationHashesRef.current = nextHashes;
    if (changedRecords.length === 0) {
      return;
    }
    void Promise.allSettled(
      changedRecords.map((record) => apiClient.upsertConversation(record))
    );
  }, [apiClient, authStatus, conversations, conversationsHydrated]);

  useEffect(() => {
    if (authStatus !== "authenticated" || !conversationsHydrated) {
      return;
    }
    const timeoutId = window.setTimeout(flushConversationSnapshots, 250);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [authStatus, conversationsHydrated, flushConversationSnapshots]);

  // Belt-and-suspenders: persist the latest snapshot synchronously when the tab
  // is hidden or the page is being unloaded, so the user's just-sent message is
  // durable before they navigate away (closing the debounce-window gap).
  const flushConversationSnapshotsRef = useRef(flushConversationSnapshots);
  useEffect(() => {
    flushConversationSnapshotsRef.current = flushConversationSnapshots;
  }, [flushConversationSnapshots]);
  useEffect(() => {
    const flush = () => flushConversationSnapshotsRef.current();
    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        flush();
      }
    };
    window.addEventListener("pagehide", flush);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      window.removeEventListener("pagehide", flush);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

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
    const resourceListParams: ResourceListRequestParams = {
      collectionId: activeResourceCollection?.collection_id ?? "",
      query: debouncedResourceQuery.trim(),
      kind: resourceKindFilter,
      source: resourceSourceFilter,
      sharing: resourceSharingFilter,
      status: resourceStatusFilter,
      tags: parseResourceTagFilter(resourceTagFilter),
      refreshToken: resourceRefreshToken,
    };
    const activeResourceListKey = buildResourceListKey(resourceListParams);
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
    const request = buildResourceListRequest(apiClient, resourceListParams, 0);
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
        setAdminMetrics(null);
        setAdminLoadingMetrics(false);
      });
    }
    let cancelled = false;
    const cancelQueuedReset = queueEffectUpdate(() => {
      if (cancelled) {
        return;
      }
      setAdminLoadingMetrics(true);
    });
    void loadAdminMetrics(apiClient, adminMetricsRangeDays)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setAdminMetrics(payload);
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
        setAdminLoadingMetrics(false);
      });
    return () => {
      cancelled = true;
      cancelQueuedReset();
    };
  }, [activePanel, adminMetricsRangeDays, adminRefreshToken, apiClient, authIsAdmin, authStatus]);

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

  // Single entry point for attaching files to the active conversation — used
  // by the composer FileUpload (pickers + card drops) and the window-level
  // chat drop overlay, so every path shares identical pending-file semantics.
  const attachFilesToActiveConversation = useCallback(
    (files: File[]): void => {
      if (files.length === 0) {
        return;
      }
      // The per-drop cap protects a single traversal, but drops accumulate in
      // pendingFiles and the server rejects sessions past the same limit at
      // send time — refuse the overage here, where the user can still react.
      const alreadyPending = activeConversation?.pendingFiles.length ?? 0;
      if (alreadyPending + files.length > MAX_DROPPED_FILES) {
        showErrorToast(
          `Attachments are limited to ${MAX_DROPPED_FILES.toLocaleString()} files per message (${alreadyPending.toLocaleString()} already attached).`
        );
        return;
      }
      updateActiveConversation((conversation) => ({
        ...conversation,
        pendingFiles: [...conversation.pendingFiles, ...files],
      }));
    },
    [activeConversation, updateActiveConversation]
  );

  /* A paste that reads as data rather than prompt (see shouldAttachPastedText)
     becomes a .txt attachment instead of flooding the composer. The chip IS the
     feedback — no toast: it appears where the text would have gone, and
     removing it is the escape hatch for anyone who truly wanted the text
     inline. */
  const attachPastedText = useCallback(
    (text: string): void => {
      attachFilesToActiveConversation([pastedTextFile(text)]);
    },
    [attachFilesToActiveConversation]
  );

  // Window-level file-drag tracking: (a) preventDefault on dragover/drop so a
  // drop that misses every target never navigates the tab away from the app,
  // and (b) a depth counter (dragenter/dragleave fire once per element
  // boundary) driving the chat drop overlay without enter/leave flicker.
  const [windowFileDragActive, setWindowFileDragActive] = useState(false);
  const windowFileDragDepthRef = useRef(0);
  useEffect(() => {
    const isFileDrag = (event: DragEvent): boolean => isOsFileDrag(event.dataTransfer);
    const handleDragEnter = (event: DragEvent): void => {
      if (!isFileDrag(event)) {
        return;
      }
      windowFileDragDepthRef.current += 1;
      setWindowFileDragActive(true);
    };
    const handleDragLeave = (event: DragEvent): void => {
      if (!isFileDrag(event)) {
        return;
      }
      windowFileDragDepthRef.current = Math.max(0, windowFileDragDepthRef.current - 1);
      if (windowFileDragDepthRef.current === 0) {
        setWindowFileDragActive(false);
      }
    };
    const handleDragOver = (event: DragEvent): void => {
      if (isFileDrag(event)) {
        event.preventDefault();
      }
    };
    const handleDrop = (event: DragEvent): void => {
      windowFileDragDepthRef.current = 0;
      setWindowFileDragActive(false);
      if (isFileDrag(event)) {
        // Inner drop zones already ran (bubble order); for unhandled drops
        // this stops the browser from opening the file over the app.
        event.preventDefault();
      }
    };
    // Capture phase, deliberately: inner drop zones (composer FileUpload,
    // ResourceBrowser tiles) stopPropagation in their handlers, and React
    // delegates at #root — a bubble-phase window listener would never fire for
    // handled drops, wedging the drag state and leaving the overlay stuck over
    // the app. Capture runs before any of that. preventDefault does not stop
    // propagation, so inner handlers still receive the drop.
    window.addEventListener("dragenter", handleDragEnter, true);
    window.addEventListener("dragleave", handleDragLeave, true);
    window.addEventListener("dragover", handleDragOver, true);
    window.addEventListener("drop", handleDrop, true);
    return () => {
      window.removeEventListener("dragenter", handleDragEnter, true);
      window.removeEventListener("dragleave", handleDragLeave, true);
      window.removeEventListener("dragover", handleDragOver, true);
      window.removeEventListener("drop", handleDrop, true);
    };
  }, []);

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

  const [welcomeNonce, setWelcomeNonce] = useState(0);
  const createNewConversation = useCallback((): void => {
    // Advance the rotating welcome prompt on every new-chat action, even when a blank
    // draft is reused (so the prompt still changes when the user clicks New chat).
    setWelcomeNonce((value) => value + 1);
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

  // Type-to-focus: start typing anywhere in a chat and the composer takes it.
  useEffect(() => {
    if (authStatus !== "authenticated" || activePanel !== "chat" || viewerOpen) {
      return;
    }
    const handleTypeToFocus = (event: KeyboardEvent): void => {
      if (!shouldTypingFocusComposer(event)) {
        return;
      }
      const textarea = composerTextareaRef.current;
      if (!textarea || textarea.disabled || textarea === document.activeElement) {
        return;
      }
      if (isEditableEventTarget(event.target) || hasBlockingOverlay()) {
        return;
      }
      // Focus SYNCHRONOUSLY and let the event run its course. The keystroke's
      // default action inserts into whatever holds focus when it fires, which
      // is after this handler returns — so focusing here means the character
      // lands in the composer on its own.
      //
      // Deliberately no preventDefault and no manual insertion: doing either
      // would either drop the character or type it twice. This is also why
      // focusComposerTextarea is not reused — it focuses inside a
      // requestAnimationFrame, a whole frame too late to catch the keystroke.
      textarea.focus({ preventScroll: true });
      const caret = textarea.value.length;
      textarea.setSelectionRange(caret, caret);
    };
    window.addEventListener("keydown", handleTypeToFocus);
    return () => window.removeEventListener("keydown", handleTypeToFocus);
  }, [activePanel, authStatus, viewerOpen]);


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
        // Keep the turn (even with no streamed text) and mark it "stopped" so the
        // user sees a calm Retry / Edit affordance instead of a silent dead-end.
        const messages = assistantMessageId
          ? conversation.messages.map((message) => {
              if (message.id !== assistantMessageId) {
                return message;
              }
              const preservedText = partialText || message.content.trim();
              return {
                ...message,
                content: preservedText,
                liveStream: undefined,
                status: "stopped" as const,
                errorReason: undefined,
              };
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
      const controller = activeChatAbortControllersRef.current.get(normalizedConversationId);
      if (controller && !controller.signal.aborted) {
        controller.abort();
      }
    },
    []
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
    setAdminMetrics(null);
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

  const activeMessages = activeConversation?.messages ?? EMPTY_UI_MESSAGES;

  /* One version chain per report path across the conversation: re-registering
     outputs/report.html on a later run appends a version behind the same card
     instead of minting a second identity. Order follows the transcript, so
     the last entry is always the latest registration. */
  const reportCanvasVersionsByKey = useMemo(() => {
    const byKey = new Map<string, ReportCanvasVersion[]>();
    for (const message of activeMessages) {
      if (message.role !== "assistant") {
        continue;
      }
      for (const document of message.runDocuments ?? EMPTY_RUN_DOCUMENTS) {
        if (document.kind !== "report") {
          continue;
        }
        const pathKey = runReportPathKey(document.path);
        if (!pathKey) {
          continue;
        }
        const versions = byKey.get(pathKey) ?? [];
        versions.push({
          messageId: message.id,
          runId: message.runId ?? null,
          document,
          imageArtifacts: message.runArtifacts ?? EMPTY_RUN_IMAGE_ARTIFACTS,
        });
        byKey.set(pathKey, versions);
      }
    }
    return byKey;
  }, [activeMessages]);
  const reportVersionCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    reportCanvasVersionsByKey.forEach((versions, pathKey) => {
      counts[pathKey] = versions.length;
    });
    return counts;
  }, [reportCanvasVersionsByKey]);
  const activeReportCanvasVersions = useMemo(() => {
    if (
      !reportCanvasTarget ||
      reportCanvasTarget.conversationId !== (activeConversation?.id ?? null)
    ) {
      return null;
    }
    const versions = reportCanvasVersionsByKey.get(reportCanvasTarget.pathKey);
    return versions && versions.length > 0 ? versions : null;
  }, [reportCanvasTarget, activeConversation?.id, reportCanvasVersionsByKey]);
  const reportCanvasVisible = Boolean(activeReportCanvasVersions);
  const openReportPathKey =
    reportCanvasVisible && !reportCanvasClosing && reportCanvasTarget
      ? reportCanvasTarget.pathKey
      : null;

  const clearReportCanvasCloseTimer = useCallback(() => {
    if (reportCanvasCloseTimerRef.current !== null) {
      window.clearTimeout(reportCanvasCloseTimerRef.current);
      reportCanvasCloseTimerRef.current = null;
    }
  }, []);
  const restoreSidebarAfterReportCanvas = useCallback(() => {
    setSidebarOpenBeforeCanvas(null);
    if (sidebarOpenBeforeCanvas === true) {
      /* Restore only if the rail is still collapsed — if the reader re-opened
         the sidebar themselves while reading, that choice stands. */
      setSidebarOpen((current) => (current === false ? true : current));
    }
  }, [sidebarOpenBeforeCanvas]);
  const openReportCanvas = useCallback(
    (conversationId: string, pathKey: string) => {
      clearReportCanvasCloseTimer();
      setReportCanvasClosing(false);
      setReportCanvasTarget({ conversationId, pathKey });
      if (reportCanvasMode === "split") {
        /* Latch what the sidebar was before the collapse, once per canvas
           session — reopening a different report while already split keeps
           the ORIGINAL pre-canvas state. */
        setSidebarOpenBeforeCanvas((latch) =>
          latch === null ? sidebarOpenRef.current : latch
        );
        setSidebarOpen(false);
      }
    },
    [clearReportCanvasCloseTimer, reportCanvasMode]
  );
  const closeReportCanvas = useCallback(() => {
    restoreSidebarAfterReportCanvas();
    const finalize = () => {
      setReportCanvasTarget(null);
      setReportCanvasClosing(false);
    };
    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reportCanvasMode !== "split" || reduceMotion) {
      clearReportCanvasCloseTimer();
      finalize();
      return;
    }
    /* Split mode animates shut: the closing attribute drives the grid column
       back to zero, and the target clears after the 220ms gesture lands. */
    setReportCanvasClosing(true);
    clearReportCanvasCloseTimer();
    reportCanvasCloseTimerRef.current = window.setTimeout(() => {
      reportCanvasCloseTimerRef.current = null;
      finalize();
    }, 260);
  }, [clearReportCanvasCloseTimer, reportCanvasMode, restoreSidebarAfterReportCanvas]);
  const toggleReportDocument = useCallback(
    (document: RunDocumentArtifact) => {
      const conversationId = activeConversation?.id;
      if (!conversationId) {
        return;
      }
      const pathKey = runReportPathKey(document.path);
      if (!pathKey) {
        return;
      }
      if (
        reportCanvasTarget &&
        !reportCanvasClosing &&
        reportCanvasTarget.conversationId === conversationId &&
        reportCanvasTarget.pathKey === pathKey
      ) {
        closeReportCanvas();
        return;
      }
      openReportCanvas(conversationId, pathKey);
    },
    [
      activeConversation?.id,
      closeReportCanvas,
      openReportCanvas,
      reportCanvasClosing,
      reportCanvasTarget,
    ]
  );

  /* Auto-open through a ref so the artifact-hydration callback (a stable
     useCallback with narrow deps) never has to depend on canvas state. */
  useEffect(() => {
    reportCanvasAutoOpenRef.current = (conversationId, pathKeys) => {
      if (typeof window === "undefined") {
        return;
      }
      /* Split only: never steal a phone screen or a narrow stage, and never
         replace a canvas the reader already has open. Measured on the stage
         shell, the same axis the regime uses. */
      const stageWidth = mainShellWidthRef.current;
      if (stageWidth === null || stageWidth < REPORT_CANVAS_SPLIT_MIN_STAGE) {
        return;
      }
      if (conversationId !== (activeConversation?.id ?? null)) {
        return;
      }
      if (reportCanvasTarget) {
        return;
      }
      const fresh = pathKeys.find(
        (pathKey) =>
          pathKey &&
          !reportCanvasAutoOpenedKeysRef.current.has(`${conversationId}::${pathKey}`)
      );
      if (!fresh) {
        return;
      }
      reportCanvasAutoOpenedKeysRef.current.add(`${conversationId}::${fresh}`);
      openReportCanvas(conversationId, fresh);
    };
  });

  /* Esc closes the canvas — unless something closer to the keyboard (dialog,
     menu, the composer's resource picker) already claimed the keypress. */
  useEffect(() => {
    if (!reportCanvasVisible) {
      return undefined;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) {
        return;
      }
      closeReportCanvas();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [reportCanvasVisible, closeReportCanvas]);

  /* Switching conversations closes the canvas without animation; a target
     whose report vanished (message deleted) clears the same way. Adjusted
     DURING RENDER — React's sanctioned reset-state-from-props pattern — so no
     effect fires setState synchronously. Idempotent under a StrictMode
     double-render: the second pass sees the target already null. A close
     timer that outlives this reset finalizes into a no-op. */
  if (
    reportCanvasTarget &&
    (reportCanvasTarget.conversationId !== (activeConversation?.id ?? null) ||
      !reportCanvasVersionsByKey.has(reportCanvasTarget.pathKey))
  ) {
    setReportCanvasTarget(null);
    if (reportCanvasClosing) {
      setReportCanvasClosing(false);
    }
    restoreSidebarAfterReportCanvas();
  }
  const activeConversationHydrated = activeConversation?.hydrated ?? true;
  const activePrompt =
    activeConversation &&
    Object.prototype.hasOwnProperty.call(
      composerDraftsByConversationId,
      activeConversation.id
    )
      ? composerDraftsByConversationId[activeConversation.id] ?? ""
      : activeConversation?.prompt ?? "";

  // Slim-composer threshold: the pill holds one calm line while the prompt
  // fits; the moment content wraps (or a newline arrives) it expands, and it
  // settles back when the prompt shortens. Measured with a hidden mirror
  // element — the textarea's own scrollHeight is floored by whichever
  // geometry state it is currently in, which would ratchet the composer open.
  const [composerPromptOverflows, setComposerPromptOverflows] = useState(false);
  const composerMirrorRef = useRef<HTMLDivElement | null>(null);
  const composerResizeObserverRef = useRef<ResizeObserver | null>(null);
  const measureComposerOverflow = useCallback((prompt: string) => {
    const textarea = composerTextareaRef.current;
    if (!textarea) {
      setComposerPromptOverflows(false);
      return;
    }
    if (prompt.includes("\n")) {
      setComposerPromptOverflows(true);
      return;
    }
    if (prompt.trim() === "") {
      setComposerPromptOverflows(false);
      return;
    }
    let mirror = composerMirrorRef.current;
    if (!mirror) {
      mirror = document.createElement("div");
      mirror.setAttribute("aria-hidden", "true");
      mirror.style.position = "absolute";
      mirror.style.visibility = "hidden";
      mirror.style.pointerEvents = "none";
      mirror.style.whiteSpace = "pre-wrap";
      mirror.style.top = "-9999px";
      document.body.appendChild(mirror);
      composerMirrorRef.current = mirror;
    }
    const computed = window.getComputedStyle(textarea);
    // Always measure against the SLIM content width, never the live padding.
    // Padding-left/right differ between slim and expanded, so measuring against
    // the current state would feed the decision back into its own geometry and
    // oscillate at the 1↔2-line boundary. The slim insets come from the shared
    // --composer-slim-pad-* vars (styles.css) so CSS and JS can't drift, and
    // clientWidth (content+padding) stays ~constant across states.
    const rootFontSize =
      Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize) || 16;
    const toPx = (value: string, fallbackPx: number) => {
      const trimmed = value.trim();
      const parsed = Number.parseFloat(trimmed);
      if (!Number.isFinite(parsed)) {
        return fallbackPx;
      }
      return trimmed.endsWith("rem") ? parsed * rootFontSize : parsed;
    };
    const slimPadLeft = toPx(computed.getPropertyValue("--composer-slim-pad-left"), 3.4 * rootFontSize);
    const slimPadRight = toPx(computed.getPropertyValue("--composer-slim-pad-right"), 7.5 * rootFontSize);
    mirror.style.width = `${Math.max(0, textarea.clientWidth - slimPadLeft - slimPadRight)}px`;
    mirror.style.font = computed.font;
    mirror.style.letterSpacing = computed.letterSpacing;
    mirror.style.lineHeight = computed.lineHeight;
    mirror.style.wordBreak = computed.wordBreak;
    mirror.style.overflowWrap = computed.overflowWrap;
    mirror.textContent = prompt;
    const lineHeight = Number.parseFloat(computed.lineHeight) || 22;
    setComposerPromptOverflows(mirror.offsetHeight / lineHeight >= 1.9);
  }, []);
  useEffect(() => {
    measureComposerOverflow(activePrompt);
  }, [activePrompt, measureComposerOverflow]);
  // Callback ref for the composer textarea: keeps composerTextareaRef in sync AND
  // (re)binds a ResizeObserver to the live node. Width changes (viewport resize,
  // sidebar collapse, side panel) move where a single line wraps; without a
  // re-measure the slim decision goes stale and slim's max-height:3rem clamp
  // would clip the now-wrapped second line until the next keystroke. A callback
  // ref binds exactly when the node mounts — a mount-time effect could run
  // before the textarea exists and, with stable deps, never re-subscribe.
  const attachComposerTextarea = useCallback(
    (node: HTMLTextAreaElement | null) => {
      composerTextareaRef.current = node;
      composerResizeObserverRef.current?.disconnect();
      composerResizeObserverRef.current = null;
      if (node && typeof ResizeObserver !== "undefined") {
        const observer = new ResizeObserver(() => {
          // Live textarea value, never a stale prompt closure.
          measureComposerOverflow(composerTextareaRef.current?.value ?? "");
        });
        observer.observe(node);
        composerResizeObserverRef.current = observer;
      }
    },
    [measureComposerOverflow]
  );
  useEffect(() => {
    return () => {
      composerResizeObserverRef.current?.disconnect();
      composerResizeObserverRef.current = null;
      composerMirrorRef.current?.remove();
      composerMirrorRef.current = null;
    };
  }, []);

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
  // Elapsed time for the in-flight turn. The completed-message footer shows
  // "<tokens> tokens · <elapsed>"; the composer had only the token half, which
  // left the loudest number on screen as the one users can least interpret.
  // Ticks once a second — fast enough to read as live, slow enough to stay calm.
  const [activeElapsedSeconds, setActiveElapsedSeconds] = useState(0);
  const activeRunStartedAtRef = useRef<number | null>(null);
  useEffect(() => {
    if (!activeSending) {
      activeRunStartedAtRef.current = null;
      setActiveElapsedSeconds(0);
      return;
    }
    // Anchor on the first tick of THIS run so a re-render mid-run (new tokens,
    // new events) never restarts the clock.
    if (activeRunStartedAtRef.current === null) {
      activeRunStartedAtRef.current = Date.now();
    }
    const tick = () => {
      const startedAt = activeRunStartedAtRef.current;
      if (startedAt !== null) {
        setActiveElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
      }
    };
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [activeSending]);
  // Metrics ADD to the status line, they don't replace it: the plain-language
  // "is processing" is what reassures during a long agentic turn, and it used to
  // vanish the moment the first usage event landed.
  const composerRunningLabel = useMemo(() => {
    const parts = ["BisQue Ultra is processing"];
    const elapsed = formatElapsedDuration(activeElapsedSeconds);
    if (elapsed) {
      parts.push(elapsed);
    }
    if (activeStreamingTokenUsage) {
      parts.push(`${formatTokens(activeStreamingTokenUsage.total_tokens)} tokens`);
    }
    return parts.join(" · ");
  }, [activeElapsedSeconds, activeStreamingTokenUsage]);
  // The headline total is cumulative across every model call in the turn — each
  // agentic step re-sends the conversation, so input is counted again each step.
  // Without this breakdown a number like "616K tokens" reads as conversation
  // size or context fullness, which it is not. Mirrors the message footer.
  const composerRunningTitle = useMemo(
    () =>
      activeStreamingTokenUsage
        ? `${activeStreamingTokenUsage.input_tokens.toLocaleString()} input · ${activeStreamingTokenUsage.output_tokens.toLocaleString()} output${
            activeStreamingTokenUsage.model ? ` · ${activeStreamingTokenUsage.model}` : ""
          }`
        : undefined,
    [activeStreamingTokenUsage]
  );
  const shouldShowBlankChatUsage =
    authStatus === "authenticated" &&
    activePanel === "chat" &&
    activeConversationHydrated &&
    activeMessages.length === 0;
  const blankChatUsageKey = `${authMode ?? ""}:${authUser ?? ""}`;
  const loadBlankChatUsage = useCallback(
    () => loadCurrentUserTokenUsage(365),
    [loadCurrentUserTokenUsage]
  );
  const {
    usage: blankChatTokenUsage,
    loading: blankChatUsageLoading,
    error: blankChatUsageError,
  } = useBlankChatTokenUsage({
    enabled: shouldShowBlankChatUsage,
    key: blankChatUsageKey,
    load: loadBlankChatUsage,
    normalizeError: normalizeApiError,
  });
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

  /* Fold a steer.* run event into the conversation's steering message.
     Returns true when the event was a steering event (callers skip the
     assistant-message fold). A steer from another tab materializes here from
     the event payload. Defined ABOVE every stream consumer that closes over
     it — effect dependency arrays evaluate at the call site (the App.tsx TDZ
     trap). */
  const applySteerRunEvent = useCallback(
    (conversationId: string, runEvent: RunEvent): boolean => {
      // normalizeV2RunEvent folds event_kind into event_type.
      const kind = String(runEvent.event_type || "");
      if (!kind.startsWith("steer.")) {
        return false;
      }
      const payload = (runEvent.payload ?? {}) as {
        steer_id?: unknown;
        message_id?: unknown;
        text?: unknown;
      };
      const steerId = typeof payload.steer_id === "string" ? payload.steer_id : "";
      if (!steerId) {
        return true;
      }
      const nextStatus: UiSteeringStatus =
        kind === "steer.applied" ? "applied" : kind === "steer.missed" ? "missed" : "pending";
      updateConversation(conversationId, (current) => {
        const index = current.messages.findIndex((item) => item.steerId === steerId);
        if (index >= 0) {
          const existing = current.messages[index];
          // Lifecycle only moves forward; a late steer.received replay must
          // not demote an applied message.
          if (
            existing.steering === nextStatus ||
            (existing.steering && existing.steering !== "pending")
          ) {
            return current;
          }
          const messages = [...current.messages];
          messages[index] = { ...existing, steering: nextStatus };
          return { ...current, messages };
        }
        const text = typeof payload.text === "string" ? payload.text : "";
        if (!text) {
          return current;
        }
        const message: UiMessage = {
          id:
            typeof payload.message_id === "string" && payload.message_id
              ? payload.message_id
              : `steer-${steerId}`,
          role: "user",
          content: text,
          createdAt: Date.now(),
          steering: nextStatus,
          steerId,
        };
        const messages = [...current.messages];
        let insertAt = current.streamingMessageId
          ? messages.findIndex((item) => item.id === current.streamingMessageId)
          : -1;
        if (insertAt < 0) {
          // No live stream (e.g. steer.missed after terminal): keep the
          // assistant last so Phase 0's settled detection stays true.
          for (let index = messages.length - 1; index >= 0; index -= 1) {
            if (messages[index].role === "assistant") {
              insertAt = index;
              break;
            }
          }
        }
        if (insertAt >= 0) {
          messages.splice(insertAt, 0, message);
        } else {
          messages.push(message);
        }
        return { ...current, messages };
      });
      return true;
    },
    [updateConversation]
  );

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
      // A backgrounded tab does no polling work: network and rAF are throttled there, and a hidden
      // tab needs no live trace updates. It catches up immediately on visibilitychange below.
      if (isTabHidden()) {
        return;
      }
      // A live SSE/resume stream registered for this conversation is the primary
      // runEvents writer (it delivers every durable event from its cursor). The poll
      // is purely the no-stream fallback: skipping while a controller is registered
      // removes the duplicate fetch/state-write path and the wholesale runEvents
      // replacement that races the SSE append. Cursor state stays untouched, so the
      // first un-gated tick still rebuilds the full authoritative snapshot.
      if (activeChatAbortControllersRef.current.has(conversationId)) {
        return;
      }
      try {
        const response = await listRunEvents(apiClient, runId, 200, { afterSequence });
        if (cancelled || response.events.length === 0) {
          return;
        }
        // Advance the cursor past EVERY event (incl. per-token deltas) so the next tick never
        // re-fetches them...
        afterSequence = response.events.reduce((current, event) => {
          const sequence = Math.floor(Number(event.payload?.sequence) || 0);
          return sequence > current ? sequence : current;
        }, afterSequence);
        // ...but only accumulate the durable structural events into runEvents — ephemeral deltas
        // must never enter the array (the same invariant the live SSE reducer enforces; otherwise
        // the poll re-introduces the per-token bloat the live path was fixed to avoid).
        // Steer lifecycle events route to their steering message (this poll is
        // the ONLY consumer when no stream is registered — without this, the
        // eyebrow would stick at "pending" forever on the poll-fallback path).
        const fresh: RunEvent[] = [];
        for (const event of response.events) {
          if (applySteerRunEvent(conversationId, event)) {
            continue;
          }
          if (!isEphemeralDeltaEvent(event)) {
            fresh.push(event);
          }
        }
        if (fresh.length === 0) {
          return;
        }
        // Coalesce reasoning deltas into a single accumulating event via the same reducer as the live
        // SSE path (appendRunEventCoalescing): the poll's wholesale replace neither re-inflates
        // message.runEvents with the full delta history nor drops the accumulated thinking text.
        collectedEvents = fresh.reduce<RunEvent[]>(
          (acc, event) => appendRunEventCoalescing(acc, event),
          [...collectedEvents]
        );
        const snapshot = collectedEvents;
        updateConversation(conversationId, (current) => ({
          ...current,
          messages: current.messages.map((item) =>
            item.id === messageId
              ? {
                  ...item,
                  runEvents: snapshot,
                  reasoning: reasoningTextFromRunEvents(snapshot) || item.reasoning,
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
    const stopVisibility = onVisibilityChange((hidden) => {
      if (!hidden) {
        void pollRunEvents(); // immediate catch-up when the tab returns to the foreground
      }
    });
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
      stopVisibility();
    };
	  }, [
	    activeConversation?.id,
	    activeStreamingMessageId,
	    activeStreamingRunId,
	    apiClient,
	    applySteerRunEvent,
	    updateConversation,
	  ]);

  const viewerUploadedFiles =
    resourceViewerContext?.uploadedFiles ?? activeAvailableUploadedFiles;
  const viewerBisqueLinksByFileId =
    resourceViewerContext?.bisqueLinksByFileId ?? activeBisqueLinksByFileId;

  const pendingPreviewFiles = useMemo(() => {
    // Collapse a folder-picked directory format (OME-Zarr: many member files sharing a
    // `*.zarr` top segment) into ONE chip — the backend commits them as one bundle resource.
    return groupPendingUploads(activePendingFiles).map((group) => {
      const firstFile = activePendingFiles[group.indices[0]];
      const isGrouped = group.isBundle || Boolean(group.isFolder);
      const canPreviewInBrowser =
        !isGrouped && firstFile ? supportsBrowserPreview(firstFile.name, firstFile.type) : false;
      return {
        key: isGrouped
          ? `${group.isBundle ? "bundle" : "folder"}:${group.name}`
          : `${group.name}-${group.totalBytes}-${group.indices[0]}`,
        name: group.name,
        sizeLabel: isGrouped
          ? `${group.indices.length} files · ${formatBytes(group.totalBytes)}`
          : formatBytes(group.totalBytes),
        canPreviewInBrowser,
        isScientific: group.isBundle || (firstFile ? isScientificUpload(firstFile.name) : false),
        isBundle: group.isBundle,
        objectUrl: canPreviewInBrowser && firstFile ? URL.createObjectURL(firstFile) : null,
        indices: group.indices,
      };
    });
  }, [activePendingFiles]);

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
  const slashWorkflowQuery = slashWorkflowSearchQuery(activePrompt);
  const filteredSlashWorkflows = useMemo(
    () => composerWorkflows?.filterComposerWorkflows(slashWorkflowQuery) ?? [],
    [composerWorkflows, slashWorkflowQuery]
  );
  const slashWorkflowGroups = useMemo<ComposerWorkflowGroup[]>(() => {
    if (!composerWorkflows) {
      return [];
    }
    const grouped = new Map<
      ComposerWorkflowDefinition["category"],
      ComposerWorkflowDefinition[]
    >();
    filteredSlashWorkflows.forEach((workflow) => {
      const existing = grouped.get(workflow.category) ?? [];
      existing.push(workflow);
      grouped.set(workflow.category, existing);
    });
    return composerWorkflows.COMPOSER_WORKFLOW_GROUP_ORDER.map((category) => ({
      category,
      items: grouped.get(category) ?? [],
    })).filter((group) => group.items.length > 0);
  }, [composerWorkflows, filteredSlashWorkflows]);
  const slashMenuOpen =
    !composerResourcePickerOpen &&
    // Allow slash selection even when a persistent workflow such as Pro Mode
    // is already active so the menu still works as a workflow switcher.
    activePrompt.startsWith("/") &&
    activePrompt !== dismissedSlashPrompt;
  const composerSubmitDisabled =
    !activeConversationHydrated || !activePrompt.trim() || slashMenuOpen;

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

  // Synchronous reentrancy latch for uploadResourceFiles (see guard inside).
  const resourcesUploadingRef = useRef(false);
  const uploadResourceFiles = useCallback(
    async (files: File[], context?: ResourceUploadReselectionContext): Promise<void> => {
      const selectedFiles = files.filter((file) => file.size >= 0);
      // Ref, not just state: drop traversal made callers async, so two quick
      // drops can both observe stale resourcesUploading=false in their render
      // closures. The ref is checked-and-set synchronously.
      if (selectedFiles.length === 0 || resourcesUploading || resourcesUploadingRef.current) {
        return;
      }
      resourcesUploadingRef.current = true;
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
      // Bundle files (zarr stores) and plain files upload in separate
      // sessions when both are present: a bundle-bearing session's response
      // carries only bundle records, which would strip the plain files'
      // records and defeat folder placement below.
      const bundleFiles = resumeSession
        ? []
        : selectedFiles.filter((file) =>
            Boolean(bundleRootForRelativePath(file.webkitRelativePath ?? ""))
          );
      const plainFiles = resumeSession
        ? selectedFiles
        : selectedFiles.filter(
            (file) => !bundleRootForRelativePath(file.webkitRelativePath ?? "")
          );
      const mixedUpload = bundleFiles.length > 0 && plainFiles.length > 0;
      // Dropped/picked folders land as folders: group plain files by the top
      // segment of their relative path, keyed to input order (the per-file
      // upload response is index-aligned with its input). Skipped when the
      // upload already targets a collection — drops onto a folder tile or
      // inside an open folder flatten into that folder — and for zarr roots,
      // which commit as single bundle resources rather than collections.
      const folderGroupIndices = new Map<string, number[]>();
      if (!activeUploadCollectionId && !resumeFrom) {
        plainFiles.forEach((file, index) => {
          const relativePath = file.webkitRelativePath ?? "";
          if (!relativePath.includes("/")) {
            return;
          }
          const topSegment = relativePath.split("/")[0] ?? "";
          if (!topSegment) {
            return;
          }
          const existing = folderGroupIndices.get(topSegment);
          if (existing) {
            existing.push(index);
          } else {
            folderGroupIndices.set(topSegment, [index]);
          }
        });
      }
      pausedResourceUploadSessionIdsRef.current.clear();
      setResourcesUploading(true);
      setResourcesError(null);
      try {
        const uploadStartedAtMs = Date.now();
        const uploadOptions = {
          onProgress: updateResourceUploadProgress,
          resumeSession,
          pauseSignal: {
            isPaused: (sessionId: string) => pausedResourceUploadSessionIdsRef.current.has(sessionId),
          },
        };
        let plainUploaded: UploadedFileRecord[];
        let allUploaded: UploadedFileRecord[];
        if (mixedUpload) {
          const bundleResponse = await apiClient.uploadFiles(bundleFiles, uploadOptions);
          const plainResponse = await apiClient.uploadFiles(plainFiles, uploadOptions);
          plainUploaded = plainResponse.uploaded;
          allUploaded = [...bundleResponse.uploaded, ...plainResponse.uploaded];
        } else {
          const response = await apiClient.uploadFiles(selectedFiles, uploadOptions);
          plainUploaded = bundleFiles.length > 0 ? [] : response.uploaded;
          allUploaded = response.uploaded;
        }
        const uploadedFileIds = uniqueFileIds(allUploaded.map((file) => file.file_id));
        // Server-side checksum dedupe returns the EXISTING record with no
        // explicit flag — the only client-visible trace is a created_at that
        // predates this upload. Surface it so re-uploads of shared datasets
        // read as intentional reuse instead of silent weirdness.
        const reusedCount = allUploaded.filter((record) => {
          const createdAtMs = Date.parse(record.created_at ?? "");
          return Number.isFinite(createdAtMs) && createdAtMs < uploadStartedAtMs - 5000;
        }).length;
        if (reusedCount > 0) {
          showSuccessToast(
            `${reusedCount} ${reusedCount === 1 ? "file" : "files"} matched existing content — reused your ${reusedCount === 1 ? "copy" : "copies"} instead of duplicating`
          );
        }
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
        } else if (folderGroupIndices.size > 0 && plainUploaded.length === plainFiles.length) {
          // Reuse-by-name must consult the server, not the resourceCollections
          // cache: that state is scoped to the current search/status filters
          // and capped, so a filtered view would create duplicate folders.
          // (Server list is capped at 200 folders; misses degrade to creating
          // a same-named folder, never to misfiling.)
          let existingFolders: ResourceCollectionRecord[] = [];
          try {
            existingFolders = (await loadResourceFolders(apiClient, { limit: 200 })).collections;
          } catch {
            // Lookup failure only disables reuse; creation below still works.
          }
          for (const [folderName, indices] of folderGroupIndices) {
            const groupFileIds = uniqueFileIds(
              indices
                .map((index) => plainUploaded[index]?.file_id)
                .filter((fileId): fileId is string => Boolean(fileId))
            );
            if (groupFileIds.length === 0) {
              continue;
            }
            try {
              const existingCollection = existingFolders.find(
                (collection) =>
                  collection.name === folderName &&
                  String(collection.collection_type ?? "") === "folder"
              );
              const collectionId = existingCollection
                ? existingCollection.collection_id
                : (
                    await apiClient.createResourceCollection({
                      name: folderName,
                      collection_type: "folder",
                      metadata: { source: "resources_drop_folder" },
                    })
                  ).collection.collection_id;
              await apiClient.addResourcesToCollection(collectionId, groupFileIds, {
                source: "resources_drop_folder",
              });
              const plural = groupFileIds.length === 1 ? "file" : "files";
              showSuccessToast(
                existingCollection
                  ? `Added ${groupFileIds.length} ${plural} to "${folderName}"`
                  : `Created folder "${folderName}" with ${groupFileIds.length} ${plural}`
              );
            } catch (error) {
              setResourcesError(
                `Uploaded, but could not file into folder "${folderName}": ${normalizeApiError(error)}`
              );
            }
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
        resourcesUploadingRef.current = false;
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

  // Resources-panel paste-to-upload: Cmd+V with a file-bearing clipboard
  // uploads into the open folder (uploadResourceFiles defaults to the active
  // collection). Guarded so it never hijacks pastes into the search box,
  // rename fields, or open dialogs; text-only pastes are untouched.
  useEffect(() => {
    const handleResourcesPaste = (event: ClipboardEvent): void => {
      if (activePanel !== "resources" || viewerOpen) {
        return;
      }
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, [contenteditable='true'], [contenteditable='']")) {
        return;
      }
      if (document.querySelector('[role="dialog"][data-state="open"]')) {
        return;
      }
      const pastedFiles = filesFromClipboard(event.clipboardData);
      if (pastedFiles.length === 0) {
        return;
      }
      event.preventDefault();
      void uploadResourceFiles(pastedFiles);
    };
    window.addEventListener("paste", handleResourcesPaste);
    return () => window.removeEventListener("paste", handleResourcesPaste);
  }, [activePanel, uploadResourceFiles, viewerOpen]);

  // Resources-panel catch-all drop: bubble phase, deliberately — the precise
  // zones (content area, folder tiles) stopPropagation on the drops they
  // consume, so this fires only for drops that would otherwise be discarded
  // (sidebar, header). Those upload to the open folder/root instead of
  // vanishing.
  useEffect(() => {
    const handleUnhandledResourcesDrop = (event: DragEvent): void => {
      if (activePanel !== "resources" || viewerOpen || !isOsFileDrag(event.dataTransfer)) {
        return;
      }
      event.preventDefault();
      const payload = snapshotDropPayload(event.dataTransfer as DataTransfer);
      void collectDroppedFiles(payload).then((dropped) => {
        const message = summarizeDropIssues(dropped);
        if (message) {
          showErrorToast(message);
        }
        if (dropped.files.length > 0) {
          void uploadResourceFiles(dropped.files);
        }
      });
    };
    window.addEventListener("drop", handleUnhandledResourcesDrop);
    return () => window.removeEventListener("drop", handleUnhandledResourcesDrop);
  }, [activePanel, uploadResourceFiles, viewerOpen]);

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
          // Creating a folder while one is open nests it there — the server
          // validates parent ownership and rejects deleted/non-folder parents.
          parent_collection_id: activeResourceCollection?.collection_id || undefined,
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
    [activeResourceCollection, apiClient]
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
        // Chips and filters read share_summary from the list — refetch so the
        // UI never claims a stale sharing state.
        setResourceRefreshToken((value) => value + 1);
        showSuccessToast("Shared — access is live");
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
        setResourceRefreshToken((value) => value + 1);
        showSuccessToast(
          `Shared ${resourceIds.length} ${resourceIds.length === 1 ? "resource" : "resources"}`
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
        setResourceRefreshToken((value) => value + 1);
        showSuccessToast("Folder shared — files added later are covered too");
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
        setResourceRefreshToken((value) => value + 1);
        showSuccessToast("Access revoked");
        return response.grant;
      } catch (error) {
        const message = normalizeApiError(error);
        setResourcesError(message);
        throw Object.assign(new Error(message), { cause: error });
      }
    },
    [apiClient]
  );

  const searchShareTargetsFromResources = useCallback(
    async (query: string): Promise<ShareTargetRecord[]> => {
      const response = await apiClient.listShareTargets(query);
      return response.targets;
    },
    [apiClient]
  );

  const loadResourceCollectionShareGrantsFromResources = useCallback(
    async (
      collection: ResourceCollectionRecord
    ): Promise<ResourceCollectionShareGrantRecord[]> => {
      const response = await apiClient.listResourceCollectionShareGrants(
        collection.collection_id
      );
      return response.grants;
    },
    [apiClient]
  );

  const revokeResourceCollectionShareGrantFromResources = useCallback(
    async (
      collection: ResourceCollectionRecord,
      grant: ResourceCollectionShareGrantRecord
    ): Promise<ResourceCollectionShareGrantRecord> => {
      setResourcesError(null);
      try {
        const response = await apiClient.revokeResourceCollectionShareGrant(
          collection.collection_id,
          grant.grant_id
        );
        // One revoke un-shares the folder AND cascades to every inherited
        // member grant server-side — refetch so chips reflect it.
        setResourceRefreshToken((value) => value + 1);
        showSuccessToast("Folder access revoked");
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
    const resourceListParams: ResourceListRequestParams = {
      collectionId: activeResourceCollection?.collection_id ?? "",
      query: debouncedResourceQuery.trim(),
      kind: resourceKindFilter,
      source: resourceSourceFilter,
      sharing: resourceSharingFilter,
      status: resourceStatusFilter,
      tags: parseResourceTagFilter(resourceTagFilter),
      refreshToken: resourceRefreshToken,
    };
    const activeResourceListKey = buildResourceListKey(resourceListParams);
    setResourcesLoadingMore(true);
    const request = buildResourceListRequest(apiClient, resourceListParams, offset);
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

  // The figure lightbox's "Open in Lens" escape hatch: resolve the resource by id
  // and hand it to the full scientific viewer.
  useEffect(() => {
    registerLightboxOpenInLens((fileId) => {
      void apiClient
        .getResource(fileId)
        .then((resource) => openResourceInViewer(resource))
        .catch(() => undefined);
    });
    return () => registerLightboxOpenInLens(null);
    // openResourceInViewer is stable for this purpose; apiClient is the only dep.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiClient]);

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
      ? { panel: "chat", resourceFileIds: [], resourceCollectionId: null }
      : parseNavFromSearch(window.location.search)
  );
  const navRestoredRef = useRef(false);
  const lastNavKeyRef = useRef<string | null>(null);

  // Rebuild the Lens viewer context from resource file id(s) (deep link / Back / refresh).
  // Always fetches fresh by id (cheap, only on navigation) so it doesn't depend on the
  // in-memory list and stays referentially stable.
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
      const uploadedFiles = uniqueByFileId(found.map(resourceToUploadedFile));
      const bisqueLinksByFileId: Record<string, BisqueViewerLink> = {};
      for (const record of found) {
        const bisqueLink = resourceToBisqueLink(record);
        if (bisqueLink) {
          bisqueLinksByFileId[record.file_id] = bisqueLink;
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
    if (initial.panel === "resources" && initial.resourceCollectionId) {
      setActiveResourceCollectionId(initial.resourceCollectionId);
    }
  }, [authStatus, restoreViewerContextForFileIds]);

  // State -> URL: push a history entry on each navigation (so Back reverses it), replace
  // on the first sync, and skip writes that originated from Back/Forward (popstate).
  useEffect(() => {
    if (typeof window === "undefined" || !navRestoredRef.current || authStatus !== "authenticated") {
      return;
    }
    const nav: NavState = {
      panel: activePanel,
      resourceFileIds: viewerResourceFileIds,
      resourceCollectionId: activePanel === "resources" ? activeResourceCollectionId : null,
    };
    const key = navStateKey(nav);
    if (key === lastNavKeyRef.current) {
      return;
    }
    const isFirstSync = lastNavKeyRef.current === null;
    lastNavKeyRef.current = key;
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
  }, [activePanel, viewerResourceFileIds, activeResourceCollectionId, authStatus]);

  // Back/Forward: restore the panel, Lens resource, Resources collection, and
  // conversation from the URL the browser navigated to. State is set to match the
  // URL, so the state->URL sync effects above see no difference and write nothing —
  // history entries are only ever created by app-driven navigation.
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const handlePopState = (): void => {
      const nav = parseNavFromSearch(window.location.search);
      // Pre-arm the dedupe key instead of a sticky suppress flag: a flag armed by
      // a pop that changes no state (chat -> chat with a different conversation)
      // would silently swallow the NEXT legitimate panel navigation. Keying makes
      // the state->URL effect a natural no-op for exactly this restore and
      // nothing else.
      lastNavKeyRef.current = navStateKey(nav);
      setActivePanel(nav.panel);
      if (nav.panel === "scientific-viewer" && nav.resourceFileIds.length > 0) {
        void restoreViewerContextForFileIds(nav.resourceFileIds);
      }
      if (nav.panel === "resources") {
        setActiveResourceCollectionId(nav.resourceCollectionId);
      }
      const urlConversationId = readConversationIdFromLocation();
      if (urlConversationId) {
        setActiveConversationId((current) => {
          if (current === urlConversationId) {
            return current;
          }
          void ensureConversationHydrated(urlConversationId);
          return urlConversationId;
        });
      }
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [ensureConversationHydrated, restoreViewerContextForFileIds]);

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
    if (!activeConversation || workflow.comingSoon) {
      return;
    }
    const preset = toComposerWorkflowPresetState(workflow);
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      updatedAt: Date.now(),
      composerWorkflowPreset: preset,
    }));
    setActivePromptValue(
      visiblePromptAfterComposerWorkflowSelection(workflow, activePrompt)
    );
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
      const nextIndex = cycleListIndex(currentIndex, direction, composerResources.length);
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

  const uploadPendingFiles = useCallback(async (
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
    // Bundle (zarr) and plain files upload in SEPARATE sessions: a
    // bundle-bearing session's client response carries only bundle records
    // (api.ts uploadMultipleFilesWithV2Session), which would silently drop the
    // plain files' records — and with them, the files themselves — from the
    // run. Same split the Resources upload path uses.
    const bundleFiles = pendingFilesSnapshot.filter((file) =>
      Boolean(bundleRootForRelativePath(file.webkitRelativePath ?? ""))
    );
    const plainFiles = pendingFilesSnapshot.filter(
      (file) => !bundleRootForRelativePath(file.webkitRelativePath ?? "")
    );
    let uploadedRecords: UploadedFileRecord[];
    if (bundleFiles.length > 0 && plainFiles.length > 0) {
      const bundleResponse = await apiClient.uploadFiles(bundleFiles);
      const plainResponse = await apiClient.uploadFiles(plainFiles);
      uploadedRecords = [...bundleResponse.uploaded, ...plainResponse.uploaded];
    } else {
      uploadedRecords = (await apiClient.uploadFiles(pendingFilesSnapshot)).uploaded;
    }
    const response = { uploaded: uploadedRecords };
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
  }, [apiClient, updateConversation]);

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

  const hydrateRunArtifacts = useCallback(async (
    conversationId: string,
    assistantMessageId: string,
    runId: string,
    options?: { autoOpenReport?: boolean }
  ): Promise<void> => {
    try {
      const artifactResponse = await listRunArtifacts(apiClient, runId, 2000);
      const durableArtifacts = artifactResponse.artifacts.filter(
        (artifact) =>
          Math.max(0, Number(artifact.size_bytes) || 0) > 0 &&
          isHydratableRunArtifactVisual({
            path: artifact.path,
            mime_type: artifact.mime_type,
          })
      );
      // Non-visual durable outputs (markdown report, supporting code, data tables)
      // become readable/downloadable documents in the chat instead of being dropped.
      const documentArtifacts: RunDocumentArtifact[] = artifactResponse.artifacts
        .filter(
          (artifact) =>
            Math.max(0, Number(artifact.size_bytes) || 0) > 0 &&
            isHydratableRunArtifactDocument({
              path: artifact.path,
              mime_type: artifact.mime_type,
            })
        )
        .map((artifact) => ({
          path: artifact.path,
          title: artifactDisplayName(artifact),
          downloadUrl: apiClient.artifactDownloadUrl(runId, artifact.path),
          kind: classifyRunDocumentKind(artifact.path, artifact.mime_type) ?? "document",
          mimeType: String(artifact.mime_type || "").trim() || undefined,
          sizeBytes: Math.max(0, Number(artifact.size_bytes) || 0) || undefined,
        }));
      if (durableArtifacts.length === 0 && documentArtifacts.length === 0) {
        return;
      }

      const selected = prioritizeHydratedImageArtifacts(durableArtifacts);

      /* Report paths this hydration introduces to the conversation — i.e.
         paths no OTHER message already carries. Only these may auto-open the
         canvas: a re-registration of an existing report (v2, v3…) updates its
         card quietly and never opens anything over the reader. Written from
         inside the updater (idempotent under a double-invoke) because that is
         where the pre-update conversation state is visible. */
      let freshReportPathKeys: string[] = [];
      updateConversation(conversationId, (conversation) => {
        const knownReportKeys = new Set<string>();
        conversation.messages.forEach((item) => {
          if (item.id === assistantMessageId) {
            return;
          }
          (item.runDocuments ?? []).forEach((document) => {
            if (document.kind === "report") {
              knownReportKeys.add(runReportPathKey(document.path));
            }
          });
        });
        freshReportPathKeys = documentArtifacts
          .filter((document) => document.kind === "report")
          .map((document) => runReportPathKey(document.path))
          .filter((pathKey) => pathKey && !knownReportKeys.has(pathKey));
        const uploadedPreviewLookup = buildUploadedArtifactPreviewLookup(
          conversation.uploadedFiles
        );
        return {
          ...conversation,
          messages: conversation.messages.map((item) =>
            item.id === assistantMessageId
              ? {
                  ...item,
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
                  runDocuments: documentArtifacts,
                }
              : item
          ),
        };
      });
      if (options?.autoOpenReport && freshReportPathKeys.length > 0) {
        reportCanvasAutoOpenRef.current?.(conversationId, freshReportPathKeys);
      }
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
    void hydrateRunArtifacts(conversationId, assistantMessageId, runId, {
      /* Live completion (not load-time backfill): the turn's first NEW report
         may open the canvas, once, on the desktop split only. */
      autoOpenReport: true,
    });
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
        // No hydrated artifacts yet: shouldHydrateRunArtifacts returns true
        // unconditionally, so skip the expensive card build entirely.
        if ((message.runArtifacts ?? []).length === 0) {
          targets.push({
            conversationId: conversation.id,
            messageId: message.id,
            runId: message.runId,
            key: hydrationKey,
          });
          return;
        }
        const decisionCache = runArtifactHydrationDecisionsRef.current;
        const cached = decisionCache.get(hydrationKey);
        let decision: boolean;
        if (
          cached &&
          cached.progressEvents === message.progressEvents &&
          cached.runArtifacts === message.runArtifacts &&
          cached.responseMetadata === message.responseMetadata &&
          cached.content === message.content &&
          cached.uploadedFiles === conversation.uploadedFiles
        ) {
          decision = cached.decision;
        } else {
          const cards = buildToolResultCards(
            message.progressEvents ?? [],
            message.runArtifacts ?? [],
            conversation.uploadedFiles,
            (fileId) => apiClient.uploadPreviewUrl(fileId)
          );
          decision = shouldHydrateRunArtifacts(message, cards);
          decisionCache.set(hydrationKey, {
            progressEvents: message.progressEvents,
            runArtifacts: message.runArtifacts,
            responseMetadata: message.responseMetadata,
            content: message.content,
            uploadedFiles: conversation.uploadedFiles,
            decision,
          });
        }
        if (!decision) {
          return;
        }
        // Drop the memo for enqueued targets so post-hydration state re-evaluates.
        decisionCache.delete(hydrationKey);
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
      let candidate: UiMessage | null = null;
      for (let index = conversation.messages.length - 1; index >= 0; index -= 1) {
        const message = conversation.messages[index];
        if (
          shouldRecoverRunResultMessage(message, {
            isStreamingMessage: conversation.streamingMessageId === message.id,
            isLocalRunActive: message.runId
              ? localActiveRunIds.has(message.runId)
              : false,
          })
        ) {
          candidate = message;
          break;
        }
      }
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
  }, [authStatus, conversations, conversationsHydrated, localActiveRunIds]);


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
            // Batch resume-stream token application to one state commit per animation frame (mirrors
            // the live path's rAF batching) so a heavy-run reconnect rebuilds content in O(n), not the
            // O(n^2) re-render-and-concat-per-token the naive append caused. resumedContent is seeded
            // lazily from the message's current content so both the reload (empty) and in-session
            // (tail) resume cases append correctly; the authoritative response_text overwrites it on
            // completion (resumeSettled gates any late flush from clobbering that).
            let resumedContent: string | null = null;
            let pendingResumeText = "";
            let resumeFlushRaf = 0;
            let resumeSettled = false;
            const flushResumeText = () => {
              resumeFlushRaf = 0;
              if (resumeSettled || controller.signal.aborted || !pendingResumeText) {
                pendingResumeText = "";
                return;
              }
              const chunk = pendingResumeText;
              pendingResumeText = "";
              updateConversation(target.conversationId, (conversation) => ({
                ...conversation,
                messages: conversation.messages.map((message) => {
                  if (message.id !== target.messageId) {
                    return message;
                  }
                  const base = resumedContent ?? message.content ?? "";
                  resumedContent = base + chunk;
                  return { ...message, content: resumedContent };
                }),
              }));
            };
            const settleResume = () => {
              resumeSettled = true;
              if (resumeFlushRaf) {
                cancelAnimationFrame(resumeFlushRaf);
                resumeFlushRaf = 0;
              }
            };
            void apiClient
              .resumeRunStream(target.runId, {
                afterSequence: target.afterSequence,
                signal: controller.signal,
                onRunEvent: (runEvent) => {
                  if (applySteerRunEvent(target.conversationId, runEvent)) {
                    return;
                  }
                  updateConversation(target.conversationId, (conversation) => ({
                    ...conversation,
                    messages: conversation.messages.map((message) =>
                      message.id === target.messageId
                        ? foldRunEventIntoMessage(message, runEvent)
                        : message
                    ),
                  }));
                },
                onToken: (delta, event) => {
                  if (
                    !shouldApplyStreamToken(
                      streamTokenDeliveriesRef.current,
                      target.conversationId,
                      target.messageId,
                      event
                    )
                  ) {
                    return;
                  }
                  pendingResumeText += delta;
                  if (typeof requestAnimationFrame !== "function") {
                    flushResumeText();
                  } else if (!resumeFlushRaf) {
                    resumeFlushRaf = requestAnimationFrame(flushResumeText);
                  }
                },
              })
              .then((response) => {
                if (controller.signal.aborted) {
                  return;
                }
                settleResume();
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
                settleResume();
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
                clearStreamTokenDeliveries(
                  streamTokenDeliveriesRef.current,
                  target.conversationId,
                  target.messageId
                );
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
    applySteerRunEvent,
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

  /* Paste-to-focus: ⌘V anywhere in a chat routes the clipboard to the composer,
     the paste sibling of type-to-focus above and of the Resources-panel
     paste-to-upload. Unlike a keydown, a paste's default action targets the
     event's own target — focusing mid-flight redirects nothing — and on an
     uneditable target that default action is a no-op. So this handler does the
     work itself: clipboardData is fully readable here, files attach through the
     same path as the composer's own onPaste, and text lands in the draft (or
     becomes an attachment when it reads as data). No double-insert risk: when
     the composer IS focused this never runs (its target is editable), and when
     it is not, nothing else would have consumed the paste. */
  useEffect(() => {
    if (authStatus !== "authenticated" || activePanel !== "chat" || viewerOpen) {
      return;
    }
    const handleChatPaste = (event: ClipboardEvent): void => {
      const textarea = composerTextareaRef.current;
      if (!textarea || textarea.disabled || !event.clipboardData) {
        return;
      }
      if (isEditableEventTarget(event.target) || hasBlockingOverlay()) {
        return;
      }
      const pastedFiles = filesFromClipboard(event.clipboardData);
      if (pastedFiles.length > 0) {
        event.preventDefault();
        attachFilesToActiveConversation(pastedFiles);
        return;
      }
      const pastedText = event.clipboardData.getData("text/plain");
      if (!pastedText) {
        return;
      }
      event.preventDefault();
      if (shouldAttachPastedText(pastedText)) {
        attachPastedText(pastedText);
      } else {
        setActivePromptValue((previous) => {
          if (!previous) {
            return pastedText;
          }
          // A paste aimed at nothing in particular lands after the draft; a
          // newline keeps it from gluing onto the last drafted word.
          return /\s$/.test(previous) ? `${previous}${pastedText}` : `${previous}\n${pastedText}`;
        });
      }
      // rAF focus is fine here (unlike type-to-focus): there is no default
      // action left to catch, and it runs after React commits the new draft,
      // so the caret lands at the true end.
      focusComposerTextarea();
    };
    window.addEventListener("paste", handleChatPaste);
    return () => window.removeEventListener("paste", handleChatPaste);
  }, [
    activePanel,
    attachFilesToActiveConversation,
    attachPastedText,
    authStatus,
    focusComposerTextarea,
    setActivePromptValue,
    viewerOpen,
  ]);

  /* Ask-about-selection: highlight transcript text, get a quiet chip that
     quotes it into the composer. The text is captured at show time, not at
     click time — the chip's own mousedown would otherwise collapse the
     selection before click fires (belt: stored text; braces: the chip also
     preventDefaults its mousedown). */
  const [selectionAsk, setSelectionAsk] = useState<{
    text: string;
    x: number;
    y: number;
  } | null>(null);

  const askAboutSelection = useCallback((): void => {
    const ask = selectionAsk;
    setSelectionAsk(null);
    if (!ask) {
      return;
    }
    /* Collapse the DOM selection the moment it is consumed. Review-confirmed:
       without this, the click's own mouseup (and Escape's keyup) re-measured
       the still-live selection and resurrected the chip a frame after every
       dismissal — and the grey inactive highlight lingered over the quoted
       text. One line fixes all three. */
    window.getSelection()?.removeAllRanges();
    // A selection large enough to read as data gets the same treatment as a
    // large paste: attachment chip, not a hundred quoted lines.
    if (shouldAttachPastedText(ask.text)) {
      attachPastedText(ask.text);
    } else {
      // Markdown blockquote, visible and editable — the user sees exactly the
      // context the model will see, and can trim it line by line.
      setActivePromptValue((previous) => draftWithQuotedSelection(previous, ask.text));
    }
    focusComposerTextarea();
  }, [attachPastedText, focusComposerTextarea, selectionAsk, setActivePromptValue]);

  useEffect(() => {
    if (
      authStatus !== "authenticated" ||
      activePanel !== "chat" ||
      viewerOpen ||
      // Fine pointers only: on touch the OS selection callout owns this
      // interaction, and a 30px chip would fight it while breaking the 44px
      // target law. Gating the whole effect keeps touch entirely native.
      !window.matchMedia("(pointer: fine)").matches
    ) {
      setSelectionAsk(null);
      return;
    }
    /* Reads the live selection and produces the chip's anchor, or null.
       Restricted to `.pk-message` at BOTH ends so sidebar titles, composer
       internals and stray UI text never grow a chip. */
    const measureSelection = (): { text: string; x: number; y: number } | null => {
      // A dialog above the transcript, or a composer that cannot accept a
      // quote yet (conversation still hydrating), means no offer.
      if (hasBlockingOverlay() || composerTextareaRef.current?.disabled) {
        return null;
      }
      const selection = window.getSelection();
      if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
        return null;
      }
      /* KaTeX-aware: a quoted formula carries its actual TeX source
         ($O(n^3)$), not the visible glyph soup — the render embeds the
         original in an annotation node, and the model reads TeX fluently. */
      const text = textFromSelection(selection);
      if (!text.trim()) {
        return null;
      }
      const withinMessage = (node: Node | null): boolean => {
        const element =
          node instanceof Element ? node : (node?.parentElement ?? null);
        return Boolean(element?.closest(".pk-message"));
      };
      if (!withinMessage(selection.anchorNode) || !withinMessage(selection.focusNode)) {
        return null;
      }
      const rect = selection.getRangeAt(0).getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) {
        return null;
      }
      // Scrolled out of the viewport: stand down instead of pinning to the
      // clamped edge over unrelated UI (review finding — the clamps below are
      // for PARTIALLY visible selections, not gone ones).
      if (
        rect.bottom < 0 ||
        rect.top > window.innerHeight ||
        rect.right < 0 ||
        rect.left > window.innerWidth
      ) {
        return null;
      }
      return {
        text,
        // Clamped so the chip never slides off-viewport on selections that
        // start at the edge; it renders translate(-50%, -100%) from this point.
        x: Math.min(Math.max(rect.left + rect.width / 2, 72), window.innerWidth - 72),
        y: Math.max(rect.top, 44),
      };
    };
    // Shown on mouseup/keyup rather than selectionchange, so the chip does not
    // flicker alongside the pointer mid-drag. selectionchange only ever hides.
    // The guards mirror dismissOnKey — review-confirmed that an unguarded
    // keyup re-showed the chip one frame after Escape dismissed it (Escape
    // does not collapse a browser selection).
    let revealFrame = 0;
    const reveal = (event: Event): void => {
      if (event instanceof KeyboardEvent) {
        if (event.key === "Escape" || isEditableEventTarget(event.target)) {
          return;
        }
      }
      if (event.target instanceof Element && event.target.closest(".chat-selection-ask")) {
        return;
      }
      if (revealFrame) {
        window.cancelAnimationFrame(revealFrame);
      }
      revealFrame = window.requestAnimationFrame(() => {
        revealFrame = 0;
        setSelectionAsk(measureSelection());
      });
    };
    const collapseWatch = (): void => {
      const selection = window.getSelection();
      if (!selection || selection.isCollapsed) {
        setSelectionAsk(null);
      }
    };
    const dismissOnKey = (event: KeyboardEvent): void => {
      // Escape dismisses outright; typing (which type-to-focus routes to the
      // composer) means the moment has passed — the highlight may linger as an
      // inactive selection, but the chip should not.
      if (event.key === "Escape" || isEditableEventTarget(event.target)) {
        setSelectionAsk(null);
      }
    };
    // Track the transcript while it scrolls under a live selection: recompute
    // from the same DOM range, rAF-throttled; hide if it left the viewport.
    let repositionFrame = 0;
    const reposition = (): void => {
      if (repositionFrame) {
        return;
      }
      repositionFrame = window.requestAnimationFrame(() => {
        repositionFrame = 0;
        setSelectionAsk((current) => (current ? measureSelection() : current));
      });
    };
    window.addEventListener("mouseup", reveal);
    window.addEventListener("keyup", reveal);
    document.addEventListener("selectionchange", collapseWatch);
    window.addEventListener("keydown", dismissOnKey);
    window.addEventListener("scroll", reposition, true);
    return () => {
      window.removeEventListener("mouseup", reveal);
      window.removeEventListener("keyup", reveal);
      document.removeEventListener("selectionchange", collapseWatch);
      window.removeEventListener("keydown", dismissOnKey);
      window.removeEventListener("scroll", reposition, true);
      if (repositionFrame) {
        window.cancelAnimationFrame(repositionFrame);
      }
      if (revealFrame) {
        window.cancelAnimationFrame(revealFrame);
      }
      setSelectionAsk(null);
    };
  }, [activePanel, authStatus, viewerOpen]);

  /* ⌘F find-within-conversation. Browser find only sees mounted DOM, and both
     transcript modes keep most of a long conversation out of the DOM (Virtuoso
     virtualization; the windowed tail behind "Show earlier messages"). So
     matching runs over the message DATA, navigation drives the virtualized
     scroller, and highlights are painted onto whichever rows exist — retried
     until the scroll has mounted the current match's row. */
  const [transcriptFindOpen, setTranscriptFindOpen] = useState(false);
  const [transcriptFindQuery, setTranscriptFindQuery] = useState("");
  const [transcriptFindIndex, setTranscriptFindIndex] = useState(0);
  const [transcriptFindNonce, setTranscriptFindNonce] = useState(0);
  /* Which conversation this find session belongs to. On a conversation switch
     the close-on-switch effect races the child transcript's scroll effect
     (child effects run first), so for one render a stale query could compute
     matches against the NEW conversation and yank its scroll. Gating on the
     captured id makes the switch render inert with no effect-ordering luck. */
  const [transcriptFindConversationId, setTranscriptFindConversationId] =
    useState<string | null>(null);
  const transcriptFindInputRef = useRef<HTMLInputElement | null>(null);

  const activeConversationIdForFind = activeConversation?.id ?? null;
  const transcriptFindActive =
    transcriptFindOpen &&
    transcriptFindConversationId === activeConversationIdForFind;

  const transcriptFindMatches = useMemo(
    () =>
      transcriptFindActive
        ? computeTranscriptFindMatches(activeMessages, transcriptFindQuery)
        : [],
    [activeMessages, transcriptFindActive, transcriptFindQuery]
  );
  /* Clamped, not reset: matches shift while an answer streams in, and a stored
     index past the end should degrade to the last match, not crash to zero. */
  const clampedTranscriptFindIndex =
    transcriptFindMatches.length > 0
      ? Math.min(transcriptFindIndex, transcriptFindMatches.length - 1)
      : 0;
  const currentTranscriptFindMatch =
    transcriptFindMatches[clampedTranscriptFindIndex] ?? null;
  const currentFindMessageId = currentTranscriptFindMatch?.messageId ?? null;
  const currentFindOccurrence = currentTranscriptFindMatch?.occurrence ?? 0;
  const currentFindMessageIndex = currentTranscriptFindMatch?.messageIndex ?? -1;
  /* Write the clamp back when it engages. Left stale, a large index silently
     re-extends when matches shrink then grow again (deletion, then a streaming
     answer), teleporting the current match with no user action. */
  useEffect(() => {
    if (transcriptFindIndex !== clampedTranscriptFindIndex) {
      setTranscriptFindIndex(clampedTranscriptFindIndex);
    }
  }, [clampedTranscriptFindIndex, transcriptFindIndex]);
  /* Keyed on primitives, NOT on the match object: matches recompute on every
     streaming delta, and a fresh object identity would re-fire the transcript's
     scroll effect and yank the reading position once per token. */
  const transcriptFindTarget = useMemo(
    () =>
      transcriptFindActive && currentFindMessageId
        ? {
            messageId: currentFindMessageId,
            /* Index travels with the id: duplicate message ids exist in real
               data (React key warnings prove it), and an id-only findIndex
               would always land on the first duplicate. */
            messageIndex: currentFindMessageIndex,
            nonce: transcriptFindNonce,
          }
        : null,
    [
      currentFindMessageId,
      currentFindMessageIndex,
      transcriptFindNonce,
      transcriptFindActive,
    ]
  );

  const openTranscriptFind = useCallback((): void => {
    // flushSync + synchronous focus, NOT an rAF: in the frame between ⌘F and a
    // deferred focus, type-to-focus would route the user's next keystrokes
    // into the composer draft. Mount the bar and take focus before the
    // handler returns; select so ⌘F with a previous query behaves like
    // browser find (type to replace, Enter to reuse).
    flushSync(() => {
      setTranscriptFindOpen(true);
      setTranscriptFindConversationId(activeConversation?.id ?? null);
    });
    transcriptFindInputRef.current?.focus();
    transcriptFindInputRef.current?.select();
  }, [activeConversation?.id]);

  const closeTranscriptFind = useCallback((): void => {
    setTranscriptFindOpen(false);
    clearTranscriptFindHighlights();
    focusComposerTextarea();
  }, [focusComposerTextarea]);

  const handleTranscriptFindQueryChange = useCallback((value: string): void => {
    setTranscriptFindQuery(value);
    // A new query restarts from its first match, live as you type.
    setTranscriptFindIndex(0);
    setTranscriptFindNonce((nonce) => nonce + 1);
  }, []);

  const goToNextTranscriptFindMatch = useCallback((): void => {
    if (transcriptFindMatches.length === 0) {
      return;
    }
    setTranscriptFindIndex(
      (clampedTranscriptFindIndex + 1) % transcriptFindMatches.length
    );
    setTranscriptFindNonce((nonce) => nonce + 1);
  }, [clampedTranscriptFindIndex, transcriptFindMatches.length]);

  const goToPreviousTranscriptFindMatch = useCallback((): void => {
    if (transcriptFindMatches.length === 0) {
      return;
    }
    setTranscriptFindIndex(
      (clampedTranscriptFindIndex - 1 + transcriptFindMatches.length) %
        transcriptFindMatches.length
    );
    setTranscriptFindNonce((nonce) => nonce + 1);
  }, [clampedTranscriptFindIndex, transcriptFindMatches.length]);

  /* ⌘F/^F opens (or refocuses) the bar. Interception is deliberate even while
     an input is focused — that is how browser find behaves — but only on the
     chat panel: Resources and the viewer render plain DOM where native find
     works, so they keep it. */
  useEffect(() => {
    if (authStatus !== "authenticated" || activePanel !== "chat" || viewerOpen) {
      return;
    }
    // ⌘F on Apple platforms, Ctrl+F elsewhere — NOT both. On macOS Ctrl+F is
    // the system-wide caret-forward binding in every text field; hijacking it
    // breaks readline muscle memory inside the composer and the find input
    // itself. event.code keeps the chord working on non-Latin layouts, where
    // event.key reports the local script.
    const isApplePlatform = /Mac|iP(hone|ad|od)/.test(window.navigator.platform);
    const handleFindShortcut = (event: KeyboardEvent): void => {
      const chordModifier = isApplePlatform
        ? event.metaKey && !event.ctrlKey
        : event.ctrlKey && !event.metaKey;
      if (
        chordModifier &&
        !event.altKey &&
        !event.shiftKey &&
        // The code fallback ONLY fires when key is not a basic Latin letter
        // (Cyrillic/Greek/Hebrew layouts). On Latin non-QWERTY layouts the
        // physical F key carries another letter — on Turkish-F, ⌘A lives on
        // KeyF, and an unconditional code match would steal select-all.
        (event.key.toLowerCase() === "f" ||
          (event.code === "KeyF" && !/^[a-z]$/i.test(event.key)))
      ) {
        if (hasBlockingOverlay()) {
          return;
        }
        // A zero-message chat has nothing for data-layer find to search, but
        // the welcome screen DOES have on-screen text — let native find keep
        // it rather than suppressing both.
        if (activeMessages.length === 0) {
          return;
        }
        event.preventDefault();
        openTranscriptFind();
      }
    };
    window.addEventListener("keydown", handleFindShortcut);
    return () => window.removeEventListener("keydown", handleFindShortcut);
  }, [activeMessages.length, activePanel, authStatus, openTranscriptFind, viewerOpen]);

  /* Switching conversations closes find: the query may be worth keeping, but a
     match list pointing into another conversation's messages is not. The
     transcriptFindActive gate above already made this render-safe; this effect
     just tidies the state. */
  useEffect(() => {
    setTranscriptFindOpen(false);
    setTranscriptFindIndex(0);
    clearTranscriptFindHighlights();
  }, [activeConversationIdForFind]);

  /* Paint highlights over mounted rows. Retries because the interesting case —
     jumping to a match far up a virtualized transcript — mounts the row some
     frames after scrollToIndex. Also re-runs on activeMessages so highlights
     track content while an answer streams. */
  useEffect(() => {
    if (!transcriptFindActive || !transcriptFindQuery.trim()) {
      clearTranscriptFindHighlights();
      return;
    }
    let cancelled = false;
    let attempts = 0;
    const apply = (): void => {
      if (cancelled) {
        return;
      }
      const { currentLocated } = applyTranscriptFindHighlights({
        query: transcriptFindQuery,
        currentMessageId: currentFindMessageId,
        currentOccurrence: currentFindOccurrence,
      });
      if (!currentLocated && attempts < 12) {
        attempts += 1;
        window.setTimeout(apply, 90);
      }
    };
    apply();
    /* Repaint as the user scrolls: Virtuoso mounts and unmounts rows, and a
       highlight registered on an unmounted row's text nodes collapses with
       them. Counts and navigation live in the data layer, so repainting is
       idempotent decoration — rAF-throttled, passive. */
    let repaintFrame = 0;
    const repaintOnScroll = (): void => {
      if (repaintFrame) {
        return;
      }
      repaintFrame = window.requestAnimationFrame(() => {
        repaintFrame = 0;
        if (!cancelled) {
          applyTranscriptFindHighlights({
            query: transcriptFindQuery,
            currentMessageId: currentFindMessageId,
            currentOccurrence: currentFindOccurrence,
          });
        }
      });
    };
    window.addEventListener("scroll", repaintOnScroll, { capture: true, passive: true });
    return () => {
      cancelled = true;
      window.removeEventListener("scroll", repaintOnScroll, true);
      if (repaintFrame) {
        window.cancelAnimationFrame(repaintFrame);
      }
    };
  }, [
    activeMessages,
    currentFindMessageId,
    currentFindOccurrence,
    transcriptFindNonce,
    transcriptFindActive,
    transcriptFindQuery,
  ]);

  /* Deleting a user message also removes every consecutive assistant reply that
     followed it (removeMessageWithPairedResponse). That is the right behaviour —
     an answer without its question is noise — but it is more than the button
     says, so the confirmation names it and the result is undoable. */
  const requestDeleteUserMessage = useCallback(
    (messageId: string): void => {
      const conversation = activeConversation;
      if (!conversation) {
        return;
      }
      const index = conversation.messages.findIndex((item) => item.id === messageId);
      if (index < 0) {
        return;
      }
      let repliesRemoved = 0;
      for (let i = index + 1; i < conversation.messages.length; i += 1) {
        if (conversation.messages[i].role !== "assistant") {
          break;
        }
        repliesRemoved += 1;
      }
      setPendingMessageDeletion({ messageId, repliesRemoved });
    },
    [activeConversation]
  );

  const cancelDeleteUserMessage = useCallback((): void => {
    setPendingMessageDeletion(null);
  }, []);

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
        /* Tombstone the runs that just went away. Without this, hydration sees
           no assistant message tagged with thread.latest_run_id, decides the
           snapshot is stale, and pushes the deleted answer back from
           control_runs.response_text — so the delete silently undoes itself on
           the next page load. Diffed against nextMessages rather than assumed,
           so only runs actually removed are recorded. */
        const survivingRunIds = new Set(
          nextMessages.map((item) => item.runId).filter(Boolean) as string[]
        );
        const removedRunIds = conversation.messages
          .map((item) => item.runId)
          .filter((runId): runId is string => typeof runId === "string" && runId.length > 0)
          .filter((runId) => !survivingRunIds.has(runId));
        return {
          ...conversation,
          updatedAt: Date.now(),
          messages: nextMessages,
          streamingMessageId: streamingRemoved ? null : activeStreamingId,
          deletedRunIds: removedRunIds.length
            ? [...new Set([...(conversation.deletedRunIds ?? []), ...removedRunIds])].slice(-200)
            : conversation.deletedRunIds ?? [],
        };
      });
    },
    [updateActiveConversation]
  );

  /* Confirmed delete. Snapshots the exact messages being removed first so Undo
     restores that set and nothing else — restoring "the last N messages" would
     resurrect the wrong ones if anything streamed in meanwhile. */
  const confirmDeleteUserMessage = useCallback((): void => {
    const pending = pendingMessageDeletion;
    const conversation = activeConversation;
    setPendingMessageDeletion(null);
    if (!pending || !conversation) {
      return;
    }
    const conversationId = conversation.id;
    const previousMessages = conversation.messages;
    const previousDeletedRunIds = conversation.deletedRunIds ?? [];
    handleDeleteUserMessage(pending.messageId);
    showUndoToast(
      pending.repliesRemoved > 0 ? "Message and its reply deleted" : "Message deleted",
      () => {
        updateConversation(conversationId, (current) => ({
          ...current,
          updatedAt: Date.now(),
          messages: previousMessages,
          // Restoring the messages must also lift their tombstones, or those run
          // ids stay permanently excluded from reconciliation.
          deletedRunIds: previousDeletedRunIds,
        }));
      }
    );
  }, [
    activeConversation,
    handleDeleteUserMessage,
    pendingMessageDeletion,
    updateConversation,
  ]);

  /* Edit a user turn.
     Previously this only wrote the composer draft: the original message stayed,
     its reply stayed, nothing re-ran, the composer was never focused, and it
     silently clobbered whatever the user had already typed. Sending afterwards
     produced a duplicated turn.

     It now means what the assistant card's "Edit" has always meant twelve lines
     below — remove the turn and its reply, put the prompt back in a focused
     composer — so the same word finally denotes the same thing in one
     transcript. Undoable, because it removes content. */
  const handleEditUserMessage = useCallback(
    (messageId: string, content: string): void => {
      const conversation = activeConversation;
      if (!conversation) {
        return;
      }
      const conversationId = conversation.id;
      const previousMessages = conversation.messages;
      const previousDeletedRunIds = conversation.deletedRunIds ?? [];
      const previousDraft = conversation.prompt;
      handleDeleteUserMessage(messageId);
      setActivePromptValue(content);
      focusComposerTextarea();
      showUndoToast("Editing — the turn was removed", () => {
        updateConversation(conversationId, (current) => ({
          ...current,
          updatedAt: Date.now(),
          messages: previousMessages,
          deletedRunIds: previousDeletedRunIds,
        }));
        setActivePromptValue(previousDraft);
      });
    },
    [
      activeConversation,
      focusComposerTextarea,
      handleDeleteUserMessage,
      setActivePromptValue,
      updateConversation,
    ]
  );

  // Retry / Edit a stopped or failed assistant turn. Both remove the stale
  // user+assistant pair so the retry is a clean re-ask rather than a duplicate;
  // "Retry" re-sends immediately (via pendingRetryPrompt), "Edit" loads the
  // prompt back into the composer for the user to adjust first.
  const retryAssistantResponse = useCallback(
    (assistantMessageId: string, options: { edit: boolean }): void => {
      const conversation = activeConversation;
      if (!conversation) {
        return;
      }
      const assistantIndex = conversation.messages.findIndex(
        (message) => message.id === assistantMessageId
      );
      if (assistantIndex < 0) {
        return;
      }
      // The originating prompt is the last NON-steering user message: a
      // mid-run steer sits between the real prompt and the assistant, and
      // taking it as the turn prompt silently re-asks the steering fragment
      // instead of the question (review-critical). The turn's steer texts
      // fold into the re-ask so nothing the user said is dropped.
      let originatingUserMessage: UiMessage | null = null;
      const turnSteerTexts: string[] = [];
      for (let index = assistantIndex; index >= 0; index -= 1) {
        const candidate = conversation.messages[index];
        if (candidate.role !== "user") {
          continue;
        }
        if (candidate.steering) {
          turnSteerTexts.unshift(candidate.content);
          continue;
        }
        originatingUserMessage = candidate;
        break;
      }
      if (!originatingUserMessage) {
        return;
      }
      const prompt = [originatingUserMessage.content, ...turnSteerTexts]
        .map((part) => part.trim())
        .filter(Boolean)
        .join("\n\n");
      const originatingUserMessageId = originatingUserMessage.id;
      const conversationId = conversation.id;
      stopRequestedConversationIdsRef.current.delete(conversationId);
      updateConversation(conversationId, (current) => ({
        ...current,
        updatedAt: Date.now(),
        messages: removeMessageWithPairedResponse(current.messages, originatingUserMessageId),
        chatError: null,
        sending: false,
        streamingMessageId: null,
      }));
      if (options.edit) {
        setActivePromptValue(prompt);
      } else {
        pendingRetryRef.current = { conversationId, prompt };
      }
    },
    [activeConversation, updateConversation, setActivePromptValue]
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

  // PERF (load-bearing): the message-row memo comparators deliberately ignore
  // `actions`, so this bag must stay referentially stable across keystrokes.
  // Every dep below must be keystroke-stable — adding a value that changes on
  // each render (e.g. the draft string) would re-render every message row.
  const transcriptActions = useMemo<ConversationTranscriptActions>(
    () => ({
      onStopConversation: stopActiveConversation,
      onStreamingRenderComplete: handleStreamingRenderComplete,
      onCopy: handleCopy,
      onOpenConversationFilesInViewer: openConversationFilesInViewer,
      onImportBisqueResourcesIntoConversation: importBisqueResourcesIntoConversation,
      onCopyBisqueResourceUri: copyBisqueResourceUri,
      onEditUserMessage: handleEditUserMessage,
      onRequestDeleteUserMessage: requestDeleteUserMessage,
      onRetryAssistant: retryAssistantResponse,
      onOpenReportDocument: toggleReportDocument,
    }),
    [
      copyBisqueResourceUri,
      handleEditUserMessage,
      requestDeleteUserMessage,
      handleCopy,
      handleStreamingRenderComplete,
      importBisqueResourcesIntoConversation,
      openConversationFilesInViewer,
      retryAssistantResponse,
      setActivePromptValue,
      stopActiveConversation,
      toggleReportDocument,
    ]
  );

  const handleSubmit = async (overridePrompt?: string): Promise<void> => {
    const conversation = activeConversation;
    if (!conversation) {
      return;
    }

    const composerWorkflowPreset = conversation.composerWorkflowPreset;
    // A retry passes the originating prompt explicitly; normal sends read the composer.
    const text = (typeof overridePrompt === "string" ? overridePrompt : activePrompt).trim();
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
    let activeLocalRunId: string | null = null;

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

      if (isChatStopRequested(conversationId)) {
        finalizeStoppedConversation({ conversationId, assistantMessageId, streamedText });
        return;
      }

      const modelPromptForTurn = composeComposerWorkflowPromptForModel(
        composerWorkflowPreset,
        promptForModel
      );
      const chatMessages = toChatWire(nextMessages);
      if (modelPromptForTurn !== text) {
        for (let idx = chatMessages.length - 1; idx >= 0; idx -= 1) {
          if (chatMessages[idx].role === "user") {
            chatMessages[idx] = { ...chatMessages[idx], content: modelPromptForTurn };
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
      // Wall-clock start, used as a fallback elapsed time when the backend run
      // record does not report duration_seconds.
      const runStartedAt = Date.now();
      const chatRequest = {
        messages: chatMessages,
        uploaded_files: [],
        file_ids: currentUploads.map((file) => file.file_id),
        conversation_id: conversationId,
        goal: modelPromptForTurn,
        selected_tool_names: effectiveSelectedToolNamesForTurn,
        remote_mutation_intents: remoteMutationIntentsForUserText(text),
        selection_context: selectionContextForTurn,
        workflow_hint: composerWorkflowPreset?.workflowHint ?? null,
        reasoning_mode: "deep" as const,
        idempotency_key: runIdempotencyKey,
      };
      chatRequestForRetry = chatRequest;

      const response = await apiClient.chatStream(chatRequest, {
        signal: chatAbortController.signal,
        onRunStarted: ({ runId }) => {
          if (!assistantMessageId || !runId) {
            return;
          }
          activeLocalRunId = runId;
          setLocalActiveRunIds((current) => {
            if (current.has(runId)) {
              return current;
            }
            const next = new Set(current);
            next.add(runId);
            return next;
          });
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
          if (applySteerRunEvent(conversationId, runEvent)) {
            return;
          }
          if (!assistantMessageId) {
            return;
          }
          updateConversation(conversationId, (current) => ({
            ...current,
            messages: current.messages.map((item) =>
              item.id === assistantMessageId
                ? foldRunEventIntoMessage(item, runEvent)
                : item
            ),
          }));
        },
        onToken: (delta, event) => {
          if (
            assistantMessageId &&
            !shouldApplyStreamToken(
              streamTokenDeliveriesRef.current,
              conversationId,
              assistantMessageId,
              event
            )
          ) {
            return;
          }
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
        (fileId) => apiClient.uploadPreviewUrl(fileId)
      );
      const responseBisqueSelection = deriveBisqueSelectionContextFromToolCards({
        toolResultCards: responseToolResultCards,
        source: "tool_result",
        originatingUserText: promptForModel,
        suggestedDomain: selectionContextForTurn?.suggested_domain ?? null,
      });
      const mergedResponseBisqueRows = responseBisqueSelection.resolvedRows;
      const mergedResponseBisqueSelectionContext = responseBisqueSelection.selectionContext;

      if (assistantMessageId) {
        const messageId = assistantMessageId;
        const elapsedSeconds = Math.max(0, (Date.now() - runStartedAt) / 1000);
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
          activeSelectionContext:
            mergedResponseBisqueSelectionContext ?? current.activeSelectionContext,
          messages: current.messages.map((item) =>
            item.id === assistantMessageId
              ? {
                ...item,
                content: assistantText,
                runId: response.run_id,
                // Prefer the backend's run duration; fall back to measured
                // wall-clock so the elapsed time is always shown.
                durationSeconds:
                  response.duration_seconds && response.duration_seconds > 0
                    ? response.duration_seconds
                    : elapsedSeconds,
                progressEvents: response.progress_events ?? [],
                runEvents: item.runEvents ?? [],
                responseMetadata: response.metadata ?? item.responseMetadata ?? null,
                liveStream: undefined,
                resolvedBisqueResources: dedupeBisqueResourceRows([
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
            (fileId) => apiClient.uploadPreviewUrl(fileId)
          );
          const fallbackBisqueSelection = deriveBisqueSelectionContextFromToolCards({
            toolResultCards: fallbackToolResultCards,
            source: "tool_result",
            originatingUserText: promptForModel,
            suggestedDomain: selectionContextForTurn?.suggested_domain ?? null,
          });
          const mergedFallbackBisqueRows = fallbackBisqueSelection.resolvedRows;
          const mergedFallbackBisqueSelectionContext = fallbackBisqueSelection.selectionContext;

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
              activeSelectionContext:
                mergedFallbackBisqueSelectionContext ?? current.activeSelectionContext,
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
                      resolvedBisqueResources: dedupeBisqueResourceRows([
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
      // Categorize the failure into a calm, actionable headline (auth / rate-limit / transport /
      // server) carrying the technical detail in parentheses, instead of surfacing a raw status
      // string. Partial streamed text is preserved as the failed message's content below.
      const failureDetail = normalizeApiError(finalError);
      const failureStatus = finalError instanceof ApiError ? finalError.status : null;
      const message = composeStreamFailureReason(
        classifyStreamFailure(failureStatus, failureDetail),
        failureDetail
      );
      if (assistantMessageId) {
        const partial = streamedText.trim();
        // Mark the turn failed and stash the technical detail (rendered in muted
        // monospace by the inline notice). The inline card is the single calm
        // error surface, so the composer-level banner is cleared.
        updateConversation(conversationId, (current) => ({
          ...current,
          updatedAt: Date.now(),
          sending: false,
          chatError: null,
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
                    content: partial,
                    liveStream: undefined,
                    status: "failed" as const,
                    errorReason: message,
                }
              : item
          ),
        }));
        return;
      }
      updateConversation(conversationId, (current) => {
        // assistantMessageId is always null here: the truthy case returned above.
        const withoutStreamingMessage = current.messages;
        return {
          ...current,
          updatedAt: Date.now(),
          sending: false,
          chatError: null,
          streamingMessageId: null,
          messages: [
            ...withoutStreamingMessage,
            {
              id: makeId(),
              role: "assistant",
              content: "",
              createdAt: Date.now(),
              status: "failed" as const,
              errorReason: message,
            },
          ],
        };
      });
    } finally {
      if (assistantMessageId) {
        clearStreamTokenDeliveries(
          streamTokenDeliveriesRef.current,
          conversationId,
          assistantMessageId
        );
      }
      if (activeLocalRunId) {
        const completedRunId = activeLocalRunId;
        setLocalActiveRunIds((current) => {
          if (!current.has(completedRunId)) {
            return current;
          }
          const next = new Set(current);
          next.delete(completedRunId);
          return next;
        });
      }
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

  // Keep a stable handle to the latest handleSubmit so the retry effect can call
  // it without re-running every render (handleSubmit is intentionally unmemoized).
  const handleSubmitRef = useRef(handleSubmit);
  useEffect(() => {
    handleSubmitRef.current = handleSubmit;
  });

  // Fire a queued "Retry" once the stale turn has been removed from state. The
  // removal re-renders with the old pair gone, so the resend appends cleanly.
  useEffect(() => {
    const pending = pendingRetryRef.current;
    if (!pending) {
      return;
    }
    if (!activeConversation || activeConversation.id !== pending.conversationId) {
      return;
    }
    if (activeConversation.sending) {
      return;
    }
    pendingRetryRef.current = null;
    void handleSubmitRef.current(pending.prompt);
  }, [activeConversation]);

  /* Conversations whose RUNNING state this session has witnessed. The
     auto-dispatch arm requires this: a reload can hydrate a failed or even
     still-active run as "settled and clean" (reconciliation rebuilds messages
     without status markers, and a transiently failed run fetch reads as
     inactive), so firing on hydrated state alone could spend a run into a
     broken context — or double-text into a live one. Unarmed conversations
     fall back to the draft-return arm: the text survives, and the user's own
     Enter is the consent. */
  const dispatchArmedConversationsRef = useRef<Set<string>>(new Set());
  const dispatchInFlightRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (activeConversation?.sending && activeConversation.id) {
      dispatchArmedConversationsRef.current.add(activeConversation.id);
    }
  }, [activeConversation?.id, activeConversation?.sending]);

  /* Queue the current draft as a follow-up to the RUNNING turn. Repeated sends
     grow the one queued message (blank-line separated) rather than stacking a
     list — a queue of N messages would auto-fire N sequential agentic runs on
     completion, which is a cost surprise nobody asked for. */
  const queueFollowup = useCallback((): void => {
    const conversation = activeConversation;
    const text = activePrompt.trim();
    // The send path refuses these; queueing must not smuggle them past it.
    if (!conversation || !text || slashMenuOpen || composerResourcePickerOpen) {
      return;
    }
    updateConversation(conversation.id, (current) => ({
      ...current,
      updatedAt: Date.now(),
      queuedFollowup: current.queuedFollowup
        ? `${current.queuedFollowup}\n\n${text}`
        : text,
    }));
    setActivePromptValue("");
  }, [activeConversation, activePrompt, setActivePromptValue, updateConversation]);

  /* Cancel returns the text to the composer (append rule, same as paste) — a
     queued thought is never destroyed, only un-queued. */
  const cancelQueuedFollowup = useCallback((): void => {
    const conversation = activeConversation;
    const queued = conversation?.queuedFollowup ?? "";
    if (!conversation || !queued) {
      return;
    }
    updateConversation(conversation.id, (current) => ({
      ...current,
      updatedAt: Date.now(),
      queuedFollowup: "",
    }));
    setActivePromptValue((previous) =>
      previous.trim() ? `${previous.replace(/\s+$/, "")}\n\n${queued}` : queued
    );
    focusComposerTextarea();
  }, [activeConversation, focusComposerTextarea, setActivePromptValue, updateConversation]);

  /* Steer the RUNNING turn (Phase 1 of double texting): the control plane
     stores the message durably and the worker folds it into the agent loop at
     its next model-call boundary — no restart, no waiting out the run. The
     steering message renders optimistically BEFORE the streaming assistant
     row (Phase 0's settled detection requires the assistant to stay last).
     A 409 steering_closed (run terminal or finalizing) falls back to the
     Phase 0 queue; any other failure returns the text to the draft — a
     steered thought is never destroyed. */
  const steerFollowup = useCallback((): void => {
    const conversation = activeConversation;
    const text = activePrompt.trim();
    if (!conversation || !text || slashMenuOpen || composerResourcePickerOpen) {
      return;
    }
    const streamingRunId = conversation.streamingMessageId
      ? conversation.messages.find((item) => item.id === conversation.streamingMessageId)?.runId
      : undefined;
    const runId = streamingRunId ?? activeRunIdByConversationRef.current.get(conversation.id);
    if (!runId) {
      // The run id has not streamed back yet — queueing is the honest option.
      queueFollowup();
      return;
    }
    const steerId = `steer_${makeId()}`;
    const conversationId = conversation.id;
    // Attachments steer WITH the text: snapshot the pending files now, upload
    // them first, and stamp their ids onto the steer itself — a steer that
    // says "use this image" while the image stays behind in the composer was
    // the exact live failure this flow replaces.
    const pendingFilesSnapshot = conversation.pendingFiles;
    const uploadedFilesSnapshot = conversation.uploadedFiles;
    const uploadedFileNames = pendingFilesSnapshot.map((file) => file.name);
    setActivePromptValue("");
    updateConversation(conversationId, (current) => {
      const message: UiMessage = {
        id: `steer-local-${steerId}`,
        role: "user",
        content: text,
        createdAt: Date.now(),
        steering: "pending",
        steerId,
        uploadedFileNames: uploadedFileNames.length > 0 ? uploadedFileNames : undefined,
      };
      const messages = [...current.messages];
      const insertAt = current.streamingMessageId
        ? messages.findIndex((item) => item.id === current.streamingMessageId)
        : -1;
      if (insertAt >= 0) {
        messages.splice(insertAt, 0, message);
      } else {
        messages.push(message);
      }
      return { ...current, updatedAt: Date.now(), messages };
    });
    void (async () => {
      let fileIds: string[] = [];
      if (pendingFilesSnapshot.length > 0) {
        try {
          const uploadResult = await uploadPendingFiles(
            conversationId,
            pendingFilesSnapshot,
            uploadedFilesSnapshot
          );
          fileIds = uploadResult.newlyUploadedFiles.map((file) => file.file_id);
        } catch (uploadError) {
          // Never send a partial steer: with the upload failed, withdraw the
          // optimistic row and hand the text back to the draft. The files are
          // still selected, so the user's retry re-sends both together.
          updateConversation(conversationId, (current) => ({
            ...current,
            updatedAt: Date.now(),
            messages: current.messages.filter((item) => item.steerId !== steerId),
            chatError:
              uploadError instanceof Error
                ? `Attachment upload failed — ${uploadError.message}`
                : "Attachment upload failed.",
          }));
          setActivePromptValue((previous) =>
            previous.trim() ? `${previous.replace(/\s+$/, "")}\n\n${text}` : text
          );
          return;
        }
      }
      return apiClient
      .steerRun(runId, { steerId, text, fileIds })
      .then((record) => {
        // Adopt the durable transcript row id so live steer.* events and the
        // next hydration converge on one message.
        updateConversation(conversationId, (current) => ({
          ...current,
          messages: current.messages.map((item) =>
            item.steerId === steerId
              ? {
                  ...item,
                  id: record.message_id || item.id,
                  steering: record.status === "applied" ? "applied" : item.steering,
                }
              : item
          ),
        }));
      })
      .catch(async (error: unknown) => {
        if (!isSteeringClosedError(error)) {
          // A lost RESPONSE is not a lost steer: the POST may have committed
          // before the network failed. Restoring the draft would tempt a
          // re-send and duplicate the transcript — verify first.
          try {
            const records = await apiClient.listRunSteerMessages(runId);
            const landed = records.find((record) => record.steer_id === steerId);
            if (landed) {
              updateConversation(conversationId, (current) => ({
                ...current,
                messages: current.messages.map((item) =>
                  item.steerId === steerId
                    ? {
                        ...item,
                        id: landed.message_id || item.id,
                        steering: landed.status === "applied" ? "applied" : item.steering,
                      }
                    : item
                ),
              }));
              return;
            }
          } catch {
            // Verification unreachable too — fall through to the restore.
          }
        }
        updateConversation(conversationId, (current) => {
          const messages = current.messages.filter((item) => item.steerId !== steerId);
          if (isSteeringClosedError(error)) {
            // Finalizing/terminal: Phase 0 owns it now — same growing-message
            // rule as queueFollowup.
            return {
              ...current,
              updatedAt: Date.now(),
              messages,
              queuedFollowup: current.queuedFollowup
                ? `${current.queuedFollowup}\n\n${text}`
                : text,
            };
          }
          return { ...current, updatedAt: Date.now(), messages };
        });
        if (!isSteeringClosedError(error)) {
          setActivePromptValue((previous) =>
            previous.trim() ? `${previous.replace(/\s+$/, "")}\n\n${text}` : text
          );
        }
      });
    })();
  }, [
    activeConversation,
    activePrompt,
    apiClient,
    composerResourcePickerOpen,
    queueFollowup,
    setActivePromptValue,
    slashMenuOpen,
    updateConversation,
    uploadPendingFiles,
  ]);

  /* Dispatch: the enqueue contract. Fires the queued follow-up as the next turn
     when the active conversation settles — but ONLY on clean completion. After
     Stop or a failure the queued text returns to the draft instead: the user
     stopped for a reason, and auto-spending a multi-minute run into a broken
     context is worse than making them press Enter. Mirrors the pendingRetry
     effect above (same settle detection, same stable submit handle); driven
     from persisted conversation state rather than a ref so a reload cannot
     lose the queue. Clearing the queue BEFORE submitting makes a double-fire
     impossible: the next effect pass sees an empty queue. */
  useEffect(() => {
    const conversation = activeConversation;
    if (!conversation || !conversation.hydrated) {
      return;
    }
    const queued = conversation.queuedFollowup.trim();
    if (!queued || conversation.sending || conversation.streamingMessageId) {
      return;
    }
    /* DEFER — do not clear — while any state that would make handleSubmit
       refuse is up. Clearing first and letting handleSubmit silently decline
       destroyed the queued text (review-confirmed: a "/"-prefixed draft
       hydrating alongside a queue was enough). These states all re-render on
       change, so the effect re-evaluates when they clear. */
    if (slashMenuOpen || composerResourcePickerOpen || conversation.selectionImportPending) {
      return;
    }
    if (dispatchInFlightRef.current.has(conversation.id)) {
      // StrictMode double-invokes effects against the same state snapshot;
      // clear-before-submit alone cannot see that.
      return;
    }
    const lastMessage = conversation.messages[conversation.messages.length - 1];
    const settled =
      lastMessage?.role === "assistant" || Boolean(conversation.chatError);
    if (!settled) {
      return;
    }
    const cleanCompletion =
      !conversation.chatError &&
      lastMessage?.role === "assistant" &&
      lastMessage.status !== "stopped" &&
      lastMessage.status !== "failed" &&
      // Only a completion this session actually WITNESSED may spend a run.
      dispatchArmedConversationsRef.current.has(conversation.id);
    dispatchInFlightRef.current.add(conversation.id);
    updateConversation(conversation.id, (current) => ({
      ...current,
      queuedFollowup: "",
    }));
    if (cleanCompletion) {
      dispatchArmedConversationsRef.current.delete(conversation.id);
      void handleSubmitRef.current(queued).finally(() => {
        dispatchInFlightRef.current.delete(conversation.id);
      });
    } else {
      dispatchInFlightRef.current.delete(conversation.id);
      setActivePromptValue((previous) =>
        previous.trim() ? `${previous.replace(/\s+$/, "")}\n\n${queued}` : queued
      );
    }
  }, [
    activeConversation,
    composerResourcePickerOpen,
    setActivePromptValue,
    slashMenuOpen,
    updateConversation,
  ]);

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
  // Memoized so typing in the composer (which re-runs the App body) doesn't
  // re-allocate/re-group the history, and so the sidebar history list keeps a
  // stable reference and its memoized rows skip reconciliation.
  const historyGroups = useMemo(
    () =>
      HISTORY_PERIOD_ORDER.map((period) => ({
        period,
        conversations: historyItems.filter((item) => item.period === period),
      })).filter((group) => group.conversations.length > 0),
    [historyItems]
  );
  /* Welcome stage: an empty, hydrated desktop chat composes hero + composer +
     starters as one centered cluster (the phone hero is already
     composer-forward, and thumb reach wants the phone composer docked). The
     flag flips off the instant the optimistic user message lands — that
     reflow IS the composer re-docking; one composer instance, no remount,
     draft and focus survive. */
  const welcomeStageActive =
    activePanel === "chat" &&
    !isPhoneView &&
    activeConversationHydrated &&
    activeMessages.length === 0;
  const welcomeStarterConversation = useMemo(
    () =>
      historyItems.find(
        (item) => item.id !== activeConversationId && item.title.trim().length > 0
      ) ?? null,
    [historyItems, activeConversationId]
  );
  const startDashboardDraft = useCallback((): void => {
    setActivePromptValue("Build an interactive dashboard from my data: ");
    focusComposerTextarea();
  }, [focusComposerTextarea, setActivePromptValue]);
  /* The hero asks a question; the keyboard should already be the answer.
     Re-fires per conversation so switching into any empty chat lands typing-
     ready, exactly like the New-chat action feels. */
  useEffect(() => {
    if (welcomeStageActive) {
      focusComposerTextarea();
    }
  }, [welcomeStageActive, activeConversationId, focusComposerTextarea]);

  // Contextual title for the mobile top bar: the panel name, or the active
  // conversation's title once it has a real exchange (else the shared app wordmark).
  const mobileShellTitle =
    activePanel === "resources"
      ? "Resources"
      : activePanel === "training"
        ? "Training"
        : activePanel === "admin"
          ? "Admin"
          : activePanel === "scientific-viewer"
            ? "Lens"
            : activeConversation &&
                activeConversation.messages.some((message) => message.role === "user")
              ? activeConversation.title
              : null;
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
      open={sidebarOpen}
      onOpenChange={setSidebarOpen}
    >
      <Sidebar collapsible="icon" className="app-sidebar">
        <CollapsedSidebarRail
          recentItems={collapsedRecentItems}
          activeConversationId={activeConversation?.id ?? null}
          resourcesActive={activePanel === "resources"}
          trainingActive={activePanel === "training"}
          lensActive={activePanel === "scientific-viewer"}
          adminActive={activePanel === "admin"}
          isAdmin={authIsAdmin}
          onCreateConversation={createNewConversation}
          onOpenResources={openResourcesPanel}
          onOpenTraining={openTrainingPanel}
          onOpenLens={openScientificViewerPanel}
          onOpenAdmin={openAdminPanel}
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
            <BrandWordmark className="app-shell-brand text-primary truncate" />
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
                    {bisqueCredentialsLinked ? (
                      <>
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
                        <SidebarMenuButton
                          className="app-bisque-link-button app-bisque-link-cta"
                          onClick={() => openSettings("bisque")}
                          title="Link your BisQue account"
                          {...mobileSidebarCloseProps}
                        >
                          <Link2 className="size-4" />
                          <span>Link BisQue account</span>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    )}
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
            {historyGroups.length === 0 && !conversationsHydrated ? (
              // Bootstrap still in flight (or failed and retriable): claiming
              // "No history yet" here reads as an empty account while the
              // list is simply not loaded yet.
              <SidebarGroup className="app-history-group">
                <SidebarGroupLabel>
                  {"Recents"}
                </SidebarGroupLabel>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton className="app-history-button" disabled>
                      <span>
                        {"Loading conversations…"}
                      </span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroup>
            ) : historyGroups.length === 0 ? (
              <SidebarGroup className="app-history-group">
                <SidebarGroupLabel>
                  {"Recents"}
                </SidebarGroupLabel>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton className="app-history-button" disabled>
                      <span>
                        {"No history yet"}
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
            onOpenSettings={() => openSettings("general")}
            onLogout={logoutBisque}
          />
        </SidebarContent>
      </Sidebar>
      {settingsDialogOpen ? (
        <Suspense fallback={null}>
          <LazyAppSettingsDialog
            open={settingsDialogOpen}
            onOpenChange={setSettingsDialogOpen}
            initialTab={settingsInitialTab}
            authUser={authUser}
            authMode={authMode}
            authIsAdmin={authIsAdmin}
            bisqueCredentialsLinked={bisqueCredentialsLinked}
            themePreference={themePreference}
            resolvedTheme={resolvedTheme}
            bisqueNavLinks={bisqueNavLinks}
            onThemePreferenceChange={setThemePreference}
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

      <SidebarInset>
        <main
          ref={setMainShellElement}
          className="app-main-shell flex min-h-0 flex-1 flex-col overflow-hidden"
          data-welcome-stage={welcomeStageActive ? "true" : undefined}
          /* Split-mode report canvas: the attribute flips the shell from a
             column into a named-area grid (bar / stage+canvas / composer),
             so the canvas gets a real column without re-nesting the chat
             tree. "open" carries the resizable column (the width variable
             below); "closing" returns it to zero and the node unmounts
             after the gesture lands. */
          data-report-canvas={
            activePanel === "chat" && reportCanvasVisible && reportCanvasMode === "split"
              ? reportCanvasClosing
                ? "closing"
                : "open"
              : undefined
          }
          style={
            activePanel === "chat" && reportCanvasVisible && reportCanvasMode === "split"
              ? ({ "--report-canvas-col": `${reportCanvasSplitWidth}px` } as CSSProperties)
              : undefined
          }
        >
          <div className="app-mobile-shell-bar md:hidden">
            <SidebarTrigger
              className="app-mobile-sidebar-trigger"
              aria-label="Open navigation"
              title="Open navigation"
            />
            <div className="app-mobile-shell-title">
              {mobileShellTitle ?? <BrandWordmark />}
            </div>
            {activePanel === "chat" ? (
              <button
                type="button"
                className="app-mobile-shell-action"
                onClick={createNewConversation}
                aria-label="New chat"
                title="New chat"
              >
                <SquarePen className="size-5" aria-hidden />
              </button>
            ) : activePanel === "resources" ? (
              // The Resources header scrolls away on mobile, so search lives here
              // in the nav bar — the one row that is always on screen.
              <button
                type="button"
                className="app-mobile-shell-action"
                onClick={() => setResourceSearchFocusSignal((signal) => signal + 1)}
                aria-label="Search resources"
                title="Search resources"
              >
                <Search className="size-5" aria-hidden />
              </button>
            ) : (
              <span className="app-mobile-shell-action-spacer" aria-hidden />
            )}
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
                metrics={adminMetrics}
                loadingMetrics={adminLoadingMetrics}
                metricsRangeDays={adminMetricsRangeDays}
                onMetricsRangeDaysChange={setAdminMetricsRangeDays}
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
                focusSearchSignal={resourceSearchFocusSignal}
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
                onSearchShareTargets={searchShareTargetsFromResources}
                onLoadCollectionShareGrants={loadResourceCollectionShareGrantsFromResources}
                onRevokeCollectionShareGrant={revokeResourceCollectionShareGrantFromResources}
                onOpenCollection={openResourceCollection}
                onRenameCollection={renameResourceCollectionFromResources}
                onDeleteCollection={deleteResourceCollectionFromResources}
                onRestoreCollection={(collection: ResourceCollectionRecord) => {
                  void restoreResourceCollection(collection);
                }}
                onClearActiveCollection={clearActiveResourceCollection}
                thumbnailUrlFor={(resource: ResourceRecord) =>
                  apiClient.resourceThumbnailUrl(resource)
                }
                zScrubThumbnail={{
                  // Gallery scrub is a transient thumbnail (never measured), so request
                  // bounded-resolution frames: the backend serves a small pyramid level
                  // for large planes, keeping rapid scrub snappy. The metadata resolved
                  // by loadZCount also keeps multichannel frames visually consistent with
                  // the static resource thumbnail instead of falling back to channel zero.
                  sliceUrlFor: (fileId: string, index: number) => {
                    const config = THUMBNAIL_SCRUB_CONFIG.get(fileId);
                    return apiClient.uploadSliceUrl(
                      fileId,
                      config
                        ? thumbnailScrubSliceRequest(config, index)
                        : { axis: "z", z: index, fullResolution: false }
                    );
                  },
                  loadZCount: (fileId: string) =>
                    apiClient.getUploadViewer(fileId).then((info) => {
                      const config = thumbnailScrubConfig(info);
                      THUMBNAIL_SCRUB_CONFIG.set(fileId, config);
                      return config.count;
                    }),
                }}
                downloadUrlFor={(resource: ResourceRecord) =>
                  apiClient.resourceDownloadUrl(resource.file_id)
                }
                collectionDownloadUrlFor={(collection: ResourceCollectionRecord) =>
                  apiClient.resourceCollectionDownloadUrl(collection.collection_id)
                }
                quickPeekFetch={(fileId: string, maxBytes: number) =>
                  apiClient.resourceTextHead(fileId, { maxBytes })
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
              />
            </Suspense>
          ) : (
            <>
            <div className="chat-stage-scroller relative min-h-0 flex-1 overflow-hidden">
              {/* Anchored to this NON-scrolling wrapper, deliberately: the
                  ChatContainerRoot below is the scroll container, and an
                  absolute child there rides the scrolled coordinate space. */}
              {transcriptFindActive ? (
                <TranscriptFindBar
                  ref={transcriptFindInputRef}
                  query={transcriptFindQuery}
                  matchCount={transcriptFindMatches.length}
                  currentIndex={clampedTranscriptFindIndex}
                  onQueryChange={handleTranscriptFindQueryChange}
                  onNext={goToNextTranscriptFindMatch}
                  onPrevious={goToPreviousTranscriptFindMatch}
                  onClose={closeTranscriptFind}
                />
              ) : null}
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
                  onScrolledAwayChange={setComposerScrolledAway}
                />
                <ConversationTranscript
                  conversationHydrated={activeConversationHydrated}
                  isPhoneView={isPhoneView}
                  welcomeName={deriveFirstName(authUser)}
                  welcomeNonce={welcomeNonce}
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
                  openReportPathKey={openReportPathKey}
                  reportVersionCounts={reportVersionCounts}
                  findTarget={transcriptFindTarget}
                />
                {/* Queued follow-up: scrolls with the transcript, below the
                    streaming answer. Not part of ConversationTranscript — its
                    memo comparator is deliberately narrow, and this is
                    conversation-level state, not a message. */}
                {activeConversationHydrated && activeConversation?.queuedFollowup ? (
                  <div className="chat-queued-followup chat-width-frame mx-auto w-full px-4 sm:px-6">
                    <div className="chat-queued-followup-bubble">
                      <div className="chat-queued-followup-eyebrow">
                        <span>Queued — sends when this run finishes</span>
                        <button
                          type="button"
                          className="chat-message-action"
                          aria-label="Cancel queued follow-up"
                          onClick={cancelQueuedFollowup}
                        >
                          <X className="size-3.5" aria-hidden="true" />
                        </button>
                      </div>
                      <div className="chat-queued-followup-text">
                        {activeConversation.queuedFollowup}
                      </div>
                    </div>
                  </div>
                ) : null}
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

          <div
            className="app-composer-shell bg-background z-10 shrink-0 px-3 pb-3 md:px-5 md:pb-5"
            /* Every width, not just phones. Reading back through a long answer
               is when the toolbar is pure distraction, and that is as true on a
               1440px screen as on a 390px one.
               Driven by scrolled-AWAY rather than actively-scrolling on purpose:
               an is-scrolling trigger would re-expand the composer the instant
               you stop to actually read a paragraph, which is exactly when you
               want it out of the way. The signal already carries hysteresis
               (collapse past 160px from the bottom, expand again within 48px),
               so it does not flutter on small scrolls. */
            data-composer-compact={
              composerScrolledAway && !activeSending ? "true" : undefined
            }
            data-composer-slim={
              !welcomeStageActive &&
              activeConversationHydrated &&
              !activeSending &&
              !composerPromptOverflows &&
              !hasComposerAttachedFiles &&
              !slashMenuOpen &&
              !composerResourcePickerOpen
                ? "true"
                : undefined
            }
            data-composer-idle={
              !welcomeStageActive &&
              activeConversationHydrated &&
              !activeSending &&
              activePrompt.trim().length === 0 &&
              !hasComposerAttachedFiles &&
              !slashMenuOpen &&
              !composerResourcePickerOpen
                ? "true"
                : undefined
            }
            data-composer-menu-open={
              slashMenuOpen || composerResourcePickerOpen ? "true" : undefined
            }
          >
            <div className="chat-width-frame mx-auto">
              {activeChatError ? (
                <SystemMessage variant="error" fill className="mb-3">
                  {activeChatError}
                </SystemMessage>
              ) : null}
              <FileUpload
                onFilesAdded={attachFilesToActiveConversation}
                onDropCollected={(collection) => {
                  const message = summarizeDropIssues(collection);
                  if (message) {
                    showErrorToast(message);
                  }
                }}
                multiple
                allowDirectories
              >
                <PromptInput
                  /* Hydration is the ONLY thing that disables typing. Folding
                     activeSending in here disabled the textarea for the whole
                     run — which made mid-run follow-ups unreachable for real
                     keyboards (review-confirmed live: disabled=true, focus() a
                     no-op; only script-dispatched events ever "worked").
                     Mid-run SUBMISSION stays blocked elsewhere: Stop replaces
                     the submit button, and handleSubmit guards sending. */
                  isLoading={!activeConversationHydrated}
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
                        workflowGroups={slashWorkflowGroups}
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
                    {/* Slim-only attach affordance: the toolbar's + collapses
                        away inside actions-start (whose opacity fade would
                        swallow any child), so the slim pill gets its own
                        trigger anchored to the card, like the idle mode echo.
                        Same Files/Folder menu as the toolbar + — one glyph,
                        one behavior. Keyboard-reachable (it IS the attach path
                        while slim); mousedown preventDefault only stops a mouse
                        click from blurring the caret. */}
                    <ComposerAttachMenu
                      variant="idle"
                      disabled={!activeConversationHydrated}
                      onCloseAutoFocus={(event) => {
                        event.preventDefault();
                        composerTextareaRef.current?.focus();
                      }}
                    />
                    {activeSending ? (
                      <div className="composer-running" title={composerRunningTitle}>
                        <Loader size="sm" text={composerRunningLabel} />
                      </div>
                    ) : null}
                    <PromptInputTextarea
                      ref={attachComposerTextarea}
                      /* The collapsed state hides the attach, model and send
                         controls, so the hint changes to say what still works.
                         "Ask Ultra" names the product next to a toolbar; with
                         the toolbar gone, an instruction is more use than a
                         label. Swaps once on collapse, not per scroll event. */
                      placeholder={
                        !activeConversationHydrated
                          ? "Loading chat…"
                          : composerScrolledAway && !activeSending
                            ? "Just start typing"
                            : "Ask Ultra"
                      }
                      /* Explicit name so the field is not relying on its
                         placeholder for one: the placeholder is deliberately
                         ghost-weight (~2.1:1) and a control's accessible name
                         should not depend on how faint its hint is drawn. */
                      aria-label="Ask Ultra"
                      className="app-composer-textarea"
                      disabled={!activeConversationHydrated}
                      onPaste={(event) => {
                        // File-bearing pastes (screenshots, Finder-copied
                        // files) attach instead of pasting; ordinary text
                        // pastes fall through untouched. Files win over rich
                        // text — paste-without-formatting is the text escape
                        // hatch.
                        const pastedFiles = filesFromClipboard(event.clipboardData);
                        if (pastedFiles.length > 0) {
                          event.preventDefault();
                          attachFilesToActiveConversation(pastedFiles);
                          return;
                        }
                        // Text that reads as data (logs, tables, sequences —
                        // see shouldAttachPastedText) becomes an attachment
                        // chip instead of burying the prompt.
                        const pastedText = event.clipboardData.getData("text/plain");
                        if (pastedText && shouldAttachPastedText(pastedText)) {
                          event.preventDefault();
                          attachPastedText(pastedText);
                        }
                      }}
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
                            const nextIndex = cycleListIndex(
                              currentIndex,
                              direction,
                              filteredSlashWorkflows.length
                            );
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
                        // ArrowUp in an EMPTY composer recalls the last prompt
                        // for refinement — shell/Slack/Discord muscle memory.
                        // Recall only, never edit: the message-level Edit
                        // removes a turn and its reply, which is far too much
                        // consequence to hang off an arrow key. Any drafted
                        // text disables it, so cursoring around a multi-line
                        // draft is untouched.
                        if (
                          event.key === "ArrowUp" &&
                          !event.nativeEvent.isComposing &&
                          !event.metaKey &&
                          !event.ctrlKey &&
                          !event.altKey &&
                          !event.shiftKey &&
                          !composerResourcePickerOpen &&
                          !activePrompt.trim()
                        ) {
                          // A pending queue outranks history: recalling the
                          // OLDER sent message while a queue exists invites
                          // queueing a duplicate. ArrowUp un-queues instead.
                          if (activeConversation?.queuedFollowup) {
                            event.preventDefault();
                            cancelQueuedFollowup();
                            return;
                          }
                          let lastPrompt = "";
                          for (let index = activeMessages.length - 1; index >= 0; index -= 1) {
                            if (activeMessages[index].role === "user") {
                              lastPrompt = activeMessages[index].content;
                              break;
                            }
                          }
                          if (lastPrompt.trim()) {
                            event.preventDefault();
                            setActivePromptValue(lastPrompt);
                            // Caret to the end once the value commits; the
                            // rAF inside runs after React paints.
                            focusComposerTextarea();
                          }
                          return;
                        }
                        // ⌘Enter during a run STEERS: the text reaches the
                        // running agent at its next model-call boundary
                        // instead of waiting out the turn (Phase 1).
                        if (
                          event.key === "Enter" &&
                          !event.shiftKey &&
                          (event.metaKey || event.ctrlKey) &&
                          !event.altKey &&
                          !event.nativeEvent.isComposing &&
                          activeSending
                        ) {
                          event.preventDefault();
                          steerFollowup();
                          return;
                        }
                        // Enter during a run queues the draft as a follow-up
                        // instead of dying against handleSubmit's sending
                        // guard. A SEPARATE branch, deliberately: the plain
                        // Enter-to-send path below is contract-pinned.
                        if (
                          event.key === "Enter" &&
                          !event.shiftKey &&
                          !event.metaKey &&
                          !event.ctrlKey &&
                          !event.altKey &&
                          !event.nativeEvent.isComposing &&
                          activeSending
                        ) {
                          event.preventDefault();
                          queueFollowup();
                          return;
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
                          {pendingPreviewFiles.map((file) => (
                            <article key={file.key} className="composer-preview-card">
                              {file.objectUrl ? (
                                <img
                                  src={file.objectUrl}
                                  alt={file.name}
                                  className="composer-preview-image"
                                />
                              ) : (
                                <div className="composer-preview-fallback">
                                  {file.isBundle ? "ZARR" : file.isScientific ? "BIO" : "FILE"}
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
                                onClick={() => {
                                  const removeIndices = new Set(file.indices);
                                  updateActiveConversation((conversation) => ({
                                    ...conversation,
                                    pendingFiles: conversation.pendingFiles.filter(
                                      (_, itemIndex) => !removeIndices.has(itemIndex)
                                    ),
                                  }));
                                }}
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
                        {/* ONE attach affordance: the browser forces two
                            hidden inputs (webkitdirectory is exclusive), but
                            the user sees a single + — format intelligence
                            (zarr re-rooting, junk, caps) lives in the shared
                            funnel, not in which button was pressed. */}
                        <ComposerAttachMenu
                          disabled={!activeConversationHydrated}
                          onCloseAutoFocus={(event) => {
                            event.preventDefault();
                            composerTextareaRef.current?.focus();
                          }}
                        />
                        <DropdownMenu>
                          <PromptInputAction
                            tooltip="Intelligence mode"
                            disabled={!activeConversationHydrated}
                            side="top"
                            sideOffset={8}
                            delayDuration={350}
                            className="app-composer-tooltip"
                          >
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
                          </PromptInputAction>
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
                        {/* The slim pill's mode control: a REAL button —
                            with the pill staying slim while typing, this is
                            the only VISIBLE mode selector (the toolbar one is
                            visibility:hidden while slim), so it must be
                            keyboard-reachable: it is a normal tab stop and its
                            slim CSS gates it out of the tab order + a11y tree
                            when the toolbar selector takes over. mousedown
                            preventDefault keeps a MOUSE click from blurring the
                            textarea; keyboard focus is unaffected. */}
                        <DropdownMenu>
                          <PromptInputAction
                            tooltip="Intelligence mode"
                            side="top"
                            sideOffset={8}
                            delayDuration={350}
                            className="app-composer-tooltip"
                          >
                            <DropdownMenuTrigger asChild>
                              <button
                                type="button"
                                className="app-composer-idle-mode"
                                data-testid="composer-slim-intelligence-trigger"
                                aria-label={`Intelligence: ${
                                  activeComposerIntelligenceMode === "pro" ? "Pro" : "High"
                                }`}
                                onMouseDown={(event) => event.preventDefault()}
                              >
                                {activeComposerIntelligenceMode === "pro" ? "Pro" : "High"}
                              </button>
                            </DropdownMenuTrigger>
                          </PromptInputAction>
                          <DropdownMenuContent
                            align="end"
                            sideOffset={8}
                            className="app-composer-intelligence-menu"
                            onCloseAutoFocus={(event) => {
                              event.preventDefault();
                              composerTextareaRef.current?.focus();
                            }}
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
                        {activeSending ? (
                          <>
                            {/* Steer, then Queue, then Stop; Stop never moves
                                (the send-position jump was a bug once
                                already). Steer reaches the RUNNING agent at
                                its next step; Queue waits the run out. */}
                            {activePrompt.trim() && !slashMenuOpen && !composerResourcePickerOpen ? (
                              <>
                                <PromptInputAction
                                  tooltip="Steer this run now — ⌘↵"
                                  side="top"
                                  sideOffset={8}
                                  delayDuration={350}
                                  className="app-composer-tooltip"
                                >
                                  <Button
                                    size="icon"
                                    type="button"
                                    variant="ghost"
                                    onClick={steerFollowup}
                                    aria-label="Steer this run"
                                    className="app-composer-steer-button size-11 rounded-full sm:size-10"
                                  >
                                    <Zap size={18} />
                                  </Button>
                                </PromptInputAction>
                                <PromptInputAction
                                  tooltip="Queue for after this run"
                                  side="top"
                                  sideOffset={8}
                                  delayDuration={350}
                                  className="app-composer-tooltip"
                                >
                                  <Button
                                    size="icon"
                                    type="button"
                                    variant="ghost"
                                    onClick={queueFollowup}
                                    aria-label="Queue follow-up"
                                    className="app-composer-queue-button size-11 rounded-full sm:size-10"
                                  >
                                    <ArrowUp size={18} />
                                  </Button>
                                </PromptInputAction>
                              </>
                            ) : null}
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
                          </>
                        ) : (
                          <PromptInputAction
                            tooltip={
                              <span className="app-composer-submit-tooltip-row">
                                <span>Send prompt</span>
                                <span
                                  className="app-composer-submit-tooltip-key"
                                  aria-hidden="true"
                                >
                                  ↵
                                </span>
                                <span className="sr-only">
                                  Press Enter to send. Shift+Enter starts a new line.
                                </span>
                              </span>
                            }
                            disabled={composerSubmitDisabled}
                            side="top"
                            sideOffset={8}
                            delayDuration={350}
                            className="app-composer-submit-tooltip"
                          >
                            <Button
                              size="icon"
                              type="submit"
                              disabled={composerSubmitDisabled}
                              aria-label="Send prompt"
                              className="app-composer-submit-button size-11 rounded-full sm:size-10"
                            >
                              <ArrowUp size={18} />
                            </Button>
                          </PromptInputAction>
                        )}
                      </div>
                    </PromptInputActions>
                  </div>
                </PromptInput>
              </FileUpload>
            </div>
          </div>
          {welcomeStageActive ? (
            <div className="welcome-starters">
              {welcomeStarterConversation ? (
                <button
                  type="button"
                  className="welcome-starter-chip"
                  onClick={() => openHistoryItem(welcomeStarterConversation)}
                >
                  <History aria-hidden="true" />
                  <span className="welcome-starter-label">
                    Continue “{welcomeStarterConversation.title}”
                  </span>
                </button>
              ) : null}
              <button
                type="button"
                className="welcome-starter-chip"
                onClick={startDashboardDraft}
              >
                <Table2 aria-hidden="true" />
                <span className="welcome-starter-label">
                  Build a dashboard from your data
                </span>
              </button>
              <button
                type="button"
                className="welcome-starter-chip"
                onClick={openScientificViewerPanel}
              >
                <Layers aria-hidden="true" />
                <span className="welcome-starter-label">Open an image in Lens</span>
              </button>
            </div>
          ) : null}
          {reportCanvasVisible && activeReportCanvasVersions ? (
            <Suspense fallback={null}>
              <LazyReportCanvas
                versions={activeReportCanvasVersions}
                mode={reportCanvasMode}
                closing={reportCanvasClosing}
                onClose={closeReportCanvas}
                loadDocumentText={fetchRunDocumentText}
                splitWidth={reportCanvasSplitWidth}
                splitWidthBounds={reportCanvasSplitBounds}
                onSplitWidthCommit={handleReportCanvasWidthCommit}
                onSplitWidthReset={handleReportCanvasWidthReset}
              />
            </Suspense>
          ) : null}
            </>
          )}
        </main>
        {/* Chat drop overlay: while an OS file drag is over the window and the
            chat panel is active, the whole viewport becomes the drop target —
            "dump files anywhere". Resources has its own zone highlighting, and
            the viewer sheet suppresses this. */}
        {windowFileDragActive &&
        activePanel === "chat" &&
        Boolean(activeConversation) &&
        activeConversationHydrated &&
        !viewerOpen ? (
          <div
            className="app-chat-drop-overlay"
            onDragOver={(event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = "copy";
            }}
            onDrop={(event) => {
              event.preventDefault();
              event.stopPropagation();
              windowFileDragDepthRef.current = 0;
              setWindowFileDragActive(false);
              const payload = snapshotDropPayload(event.dataTransfer);
              void collectDroppedFiles(payload).then((collection) => {
                attachFilesToActiveConversation(collection.files);
                const message = summarizeDropIssues(collection);
                if (message) {
                  showErrorToast(message);
                }
              });
            }}
          >
            <div className="app-chat-drop-overlay-card">
              <p className="app-chat-drop-overlay-title">Drop to attach</p>
              <p className="app-chat-drop-overlay-hint">
                Files and folders upload when you send
              </p>
            </div>
          </div>
        ) : null}
        {/* Resources drag hint: pointer-events none on purpose — the precise
            drop zones (tiles, content area) must keep receiving the drop; the
            catch-all bubble handler covers everywhere else. */}
        {windowFileDragActive && activePanel === "resources" && !viewerOpen ? (
          <div className="app-resources-drop-hint" aria-hidden="true">
            Drop anywhere to upload
            {activeResourceCollection ? ` into "${activeResourceCollection.name}"` : " to Resources"}
          </div>
        ) : null}
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
            />
          </Suspense>
        ) : null}
        {/* Ask-about-selection chip. Portaled to body: the transcript sits in
            transformed/overflow ancestors that would trap or clip a
            position:fixed child. Text was captured at show time, so the click
            works even if the browser collapses the selection first. */}
        {selectionAsk
          ? createPortal(
              <button
                type="button"
                className="chat-selection-ask"
                style={{ left: selectionAsk.x, top: selectionAsk.y }}
                onMouseDown={(event) => {
                  event.preventDefault();
                }}
                onClick={askAboutSelection}
              >
                <TextQuote className="size-3.5" aria-hidden="true" />
                Ask about this
              </button>,
              document.body
            )
          : null}
        {/* Message delete. Deliberately the same dialog grammar as conversation
            delete: the two differ in scope, not in kind, and a lighter-weight
            confirmation here would imply this one is safe to fire blind. */}
        <AlertDialog
          open={Boolean(pendingMessageDeletion)}
          onOpenChange={(open) => {
            if (!open) {
              cancelDeleteUserMessage();
            }
          }}
        >
          <AlertDialogContent size="default">
            <AlertDialogHeader>
              <AlertDialogMedia className="bg-destructive/12 text-destructive">
                <Trash className="size-7" />
              </AlertDialogMedia>
              <AlertDialogTitle>Delete this message?</AlertDialogTitle>
              <AlertDialogDescription>
                {pendingMessageDeletion && pendingMessageDeletion.repliesRemoved > 0
                  ? pendingMessageDeletion.repliesRemoved === 1
                    ? "This also removes the reply it produced."
                    : `This also removes the ${pendingMessageDeletion.repliesRemoved} replies it produced.`
                  : "This removes it from the conversation."}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel onClick={cancelDeleteUserMessage}>Cancel</AlertDialogCancel>
              <AlertDialogAction variant="destructive" onClick={confirmDeleteUserMessage}>
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
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
              {/* The strong wording is back, and now it is true: the handler
                  hard-deletes the thread row, the schema cascades take the
                  messages, runs, events and artifacts with it, and the artifact
                  blobs are unlinked afterwards. Between the audit and that
                  change this read "removed from your history", because the
                  backend was only flipping a status column. If deletion ever
                  softens again, soften this sentence in the same commit. */}
              <AlertDialogDescription>
                {`Permanently delete "${pendingConversationDelete?.title ?? "this conversation"}"? Its messages, results, and files it produced are erased. This cannot be undone.`}
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
      <FigureLightboxRoot />
    </SidebarProvider>
  );
}
