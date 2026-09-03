import {
  forwardRef,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useId,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { ArrowUp, Check, CornerDownLeft, Gauge, Slash, Square, Zap } from "lucide-react";

import { ResourceMentionPicker } from "@/components/chat/ResourceMentionPicker";
import { useFileUploadContext } from "@/components/prompt-kit/file-upload";
import { RecorderTraceIcon } from "@/components/icons/MeridianIcons";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { OneTimeNotice } from "@/components/ui/one-time-notice";
import { TooltipProvider } from "@/components/ui/tooltip";
import { resourceMentionOptionId } from "@/features/chat/resource-mention";
import type { ResourceRecord } from "@/types";

import { ComposerAttachMenu } from "./ComposerAttachMenu";
import { ComposerFallbackEditor } from "./ComposerFallbackEditor";
import { ComposerTooltip } from "./ComposerTooltip";
import type {
  ComposerEditorProps,
  ComposerFileToken,
  ComposerHandle,
  ComposerMention,
  ComposerTokenDetails,
} from "./composerHandle";
import {
  clampMentionAnchor,
  composerKeysHint,
  composerPlaceholder,
  composerSummary,
  deriveComposerStage,
} from "./composerModel";

/* The composer. One sentence that says everything the run will receive: the
   workflow and the mode lead it as chips, files sit inside it as tokens, and a
   whisper under it keeps the summary and the key contract. Its geometry is a
   function of a handful of explicit states (see composerModel) — never of a
   CSS attribute cascade — so every state is one you can name and test. */

export const loadComposerEditorModule = () => import("./ComposerEditor");

const LazyComposerEditor = lazy(() =>
  loadComposerEditorModule().then((module) => ({ default: module.ComposerEditor }))
);

export type ComposerMode = "high" | "pro";

export type ComposerWorkflowChip = {
  id: string;
  label: string;
  requiresFiles: boolean;
};

export type ComposerMentionState = {
  query: string;
  results: ResourceRecord[];
  loading: boolean;
  error?: string | null;
  activeFileId: string | null;
};

export type ComposerNotice = {
  text: string;
  action?: { label: string; onClick: () => void };
};

export type ComposerProIntro = {
  noticeId: string;
  audienceId: string;
  enabled: boolean;
  title: string;
  description: string;
};

export type ComposerProps = {
  value: string;
  onValueChange: (text: string) => void;
  tokens: readonly ComposerFileToken[];
  goneFileIds: readonly string[];
  onTokensChange: (tokens: ComposerFileToken[]) => void;
  tokenDetails: (fileId: string) => ComposerTokenDetails | null;
  onTokenRemoved?: (fileId: string) => void;

  hydrated: boolean;
  running: boolean;
  /** Scrolled away from the bottom of a long answer: the composer becomes a strip. */
  readMode: boolean;
  phone: boolean;
  welcomeStage: boolean;
  /** A slash menu or the library picker is open above the line. */
  menuOpen: boolean;
  /** Files attached by paths that do not write tokens (pending uploads, notes). */
  hasFiles: boolean;

  workflow: ComposerWorkflowChip | null;
  onChangeWorkflow: () => void;
  onRemoveWorkflow: () => void;
  mode: ComposerMode;
  onChangeMode: (mode: ComposerMode) => void;
  proIntro?: ComposerProIntro;

  notice?: ComposerNotice | null;

  mention: ComposerMentionState | null;
  onMentionChange: (mention: ComposerMention | null) => void;
  onMentionActivate: (fileId: string) => void;
  onMentionPick: (resource: ResourceRecord) => void;

  /** Slash workflows and the library picker, rendered by the app above the line. */
  menus?: ReactNode;
  /** Notes chips and pending-upload previews, rendered by the app under the line. */
  extras?: ReactNode;

  runningLabel?: string;
  runningTitle?: string;
  submitDisabled: boolean;
  /** Steer and Queue show only when there is text to send and no menu is open. */
  canSteer: boolean;
  onSubmit: () => void;
  onSteer: () => void;
  onQueue: () => void;
  onStop: () => void;

  /** The app's first look at a key: slash-menu navigation, ArrowUp recall. */
  onKeyDown?: (event: KeyboardEvent) => boolean;
  /** Files and data-shaped text attach instead of pasting. */
  onPaste?: (event: ClipboardEvent) => boolean;

  onOpenResources: () => void;
  onStartWorkflow: () => void;
  onOpenNotes?: () => void;
};

const isControlTarget = (target: EventTarget | null): boolean =>
  target instanceof Element &&
  target.closest("button, a, [role='menu'], [role='listbox'], .composer-mention-picker, .composer-menus") !==
    null;

export const Composer = forwardRef<ComposerHandle, ComposerProps>(function Composer(props, ref) {
  const {
    value,
    tokens,
    goneFileIds,
    hydrated,
    running,
    readMode,
    phone,
    welcomeStage,
    menuOpen,
    hasFiles,
    workflow,
    mode,
    notice,
    mention,
    menus,
    extras,
    runningLabel,
    runningTitle,
    submitDisabled,
    canSteer,
  } = props;

  const editorRef = useRef<ComposerHandle | null>(null);
  const fallbackRef = useRef<ComposerHandle | null>(null);
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;

  const [focused, setFocused] = useState(false);
  const focusedRef = useRef(false);
  const [editorReady, setEditorReady] = useState(false);
  const [dismissedQuery, setDismissedQuery] = useState<string | null>(null);
  const [mentionAnchor, setMentionAnchor] = useState<number | null>(null);
  const prefixRef = useRef<HTMLSpanElement | null>(null);
  const [prefixWidth, setPrefixWidth] = useState(0);
  const listboxId = useId();
  const { openFilePicker } = useFileUploadContext();

  const current = useCallback((): ComposerHandle | null => editorRef.current ?? fallbackRef.current, []);

  useImperativeHandle(
    ref,
    (): ComposerHandle => ({
      get element() {
        return current()?.element ?? null;
      },
      get disabled() {
        return current()?.disabled ?? true;
      },
      get value() {
        return current()?.value ?? propsRef.current.value;
      },
      focus: (options) => current()?.focus(options),
      isFocused: () => current()?.isFocused() ?? false,
      setValue: (text) => current()?.setValue(text),
      insertText: (text) => current()?.insertText(text),
      acceptMention: (token) => current()?.acceptMention(token),
      appendToken: (token) => current()?.appendToken(token),
      removeToken: (fileId) => current()?.removeToken(fileId),
      reopenMentionFor: (fileId) => current()?.reopenMentionFor(fileId),
      mentionRect: () => current()?.mentionRect() ?? null,
    }),
    [current]
  );

  const hasText = value.trim().length > 0;
  const stage = deriveComposerStage({
    running,
    focused,
    hasText,
    hasTokens: tokens.length > 0,
    hasFiles,
    hasWorkflow: workflow !== null,
    menuOpen,
    welcomeStage,
  });
  const collapsed = readMode && !running && !focused;
  const mentionOpen = mention !== null && dismissedQuery !== mention.query;
  const placeholder = composerPlaceholder({
    hydrated,
    welcomeStage,
    readMode: collapsed,
    running,
    hasTokens: tokens.length > 0,
    hasFiles,
    phone,
  });

  useEffect(() => {
    if (mention === null) {
      setDismissedQuery(null);
    }
  }, [mention]);

  /* Anchor the picker at the "@": measured after paint from the live editor,
     relative to the surface, clamped so the popover stays inside it. */
  useLayoutEffect(() => {
    if (!mentionOpen || phone) {
      setMentionAnchor(null);
      return;
    }
    const rect = current()?.mentionRect();
    const surface = surfaceRef.current?.getBoundingClientRect();
    if (!rect || !surface) {
      setMentionAnchor(null);
      return;
    }
    setMentionAnchor(clampMentionAnchor(rect.left - surface.left, surface.width));
  }, [mentionOpen, phone, mention?.query, current]);

  /* The placeholder sits on the first line after the chips. Their width is
     measured, never guessed, and written as a variable the sheet reads. */
  useLayoutEffect(() => {
    const node = prefixRef.current;
    if (!node) {
      setPrefixWidth(0);
      return;
    }
    const report = () => setPrefixWidth(node.getBoundingClientRect().width);
    report();
    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(report);
    observer.observe(node);
    return () => observer.disconnect();
  }, [workflow?.label, mode, hydrated]);

  const handleFocusChange = useCallback((next: boolean) => {
    focusedRef.current = next;
    setFocused(next);
  }, []);

  const handleEditorReady = useCallback(() => {
    setEditorReady(true);
    // The fallback had the caret when the real editor arrived: keep typing.
    if (focusedRef.current) {
      editorRef.current?.focus({ caret: "end", preventScroll: true });
    }
  }, []);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent): boolean => {
      const state = propsRef.current;
      const activeMention = state.mention;
      const open = activeMention !== null && dismissedQuery !== activeMention.query;
      if (open && activeMention && !event.isComposing) {
        if (event.key === "Escape") {
          event.preventDefault();
          setDismissedQuery(activeMention.query);
          return true;
        }
        if ((event.key === "ArrowDown" || event.key === "ArrowUp") && activeMention.results.length > 0) {
          event.preventDefault();
          const ids = activeMention.results.map((resource) => resource.file_id);
          const index = ids.indexOf(activeMention.activeFileId ?? "");
          const step = event.key === "ArrowDown" ? 1 : -1;
          const next = index < 0 ? (step > 0 ? 0 : ids.length - 1) : (index + step + ids.length) % ids.length;
          state.onMentionActivate(ids[next]);
          return true;
        }
        if (event.key === "Enter" || event.key === "Tab") {
          // The library is still answering: Enter must not fall through and
          // SEND a brief the picker was about to complete. It waits.
          if (activeMention.loading) {
            event.preventDefault();
            return true;
          }
          if (activeMention.results.length > 0) {
            event.preventDefault();
            const pick =
              activeMention.results.find((resource) => resource.file_id === activeMention.activeFileId) ??
              activeMention.results[0];
            state.onMentionPick(pick);
            return true;
          }
        }
      }
      return state.onKeyDown?.(event) ?? false;
    },
    [dismissedQuery]
  );

  const handleEnter = useCallback((event: KeyboardEvent): boolean => {
    const state = propsRef.current;
    event.preventDefault();
    if (!state.hydrated) {
      return true;
    }
    if (event.metaKey || event.ctrlKey) {
      if (state.running) {
        state.onSteer();
      } else if (!state.submitDisabled) {
        state.onSubmit();
      }
      return true;
    }
    if (state.running) {
      state.onQueue();
      return true;
    }
    if (!state.submitDisabled) {
      state.onSubmit();
    }
    return true;
  }, []);

  const handlePaste = useCallback((event: ClipboardEvent): boolean => {
    return propsRef.current.onPaste?.(event) ?? false;
  }, []);

  const handleTokenRemoveClick = useCallback((fileId: string) => {
    propsRef.current.onTokenRemoved?.(fileId);
  }, []);

  const handleSurfaceMouseDown = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      const handle = current();
      const element = handle?.element;
      if (!handle || !element || isControlTarget(event.target)) {
        return;
      }
      if (element.contains(event.target as Node)) {
        return;
      }
      event.preventDefault();
      handle.focus({ caret: "end" });
    },
    [current]
  );

  const activeOptionId =
    mentionOpen && mention?.activeFileId ? resourceMentionOptionId(listboxId, mention.activeFileId) : undefined;

  const editorProps: ComposerEditorProps = {
    value,
    tokens,
    goneFileIds,
    disabled: !hydrated,
    placeholder,
    ariaLabel: "Ask Ultra",
    mentionOpen,
    listboxId,
    activeOptionId,
    tokenDetails: props.tokenDetails,
    onValueChange: props.onValueChange,
    onTokensChange: props.onTokensChange,
    onMentionChange: props.onMentionChange,
    onFocusChange: handleFocusChange,
    onKeyDown: handleKeyDown,
    onEnter: handleEnter,
    onPaste: handlePaste,
    onTokenRemoveClick: handleTokenRemoveClick,
    onReady: handleEditorReady,
  };

  const summary = useMemo(
    () =>
      composerSummary({
        fileCount: tokens.length,
        workflowLabel: workflow?.label ?? null,
        modeLabel: mode === "pro" ? "Pro" : null,
      }),
    [tokens.length, workflow?.label, mode]
  );
  const whisperVisible =
    hydrated && !collapsed && (hasText || tokens.length > 0 || workflow !== null || Boolean(notice));

  const modeLabel = mode === "pro" ? "Pro" : "High";
  const modeChip = (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="composer-chip composer-chip-mode"
          aria-label={`Intelligence: ${modeLabel}`}
          data-testid="composer-mode-chip"
          disabled={!hydrated}
          onMouseDown={(event) => event.preventDefault()}
        >
          <Gauge aria-hidden="true" />
          <span>{modeLabel}</span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        sideOffset={8}
        className="app-composer-intelligence-menu"
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          current()?.focus();
        }}
      >
        <DropdownMenuLabel>Intelligence</DropdownMenuLabel>
        <DropdownMenuGroup>
          <DropdownMenuItem
            data-active={mode === "high" ? "true" : undefined}
            onClick={() => props.onChangeMode("high")}
          >
            <span>High</span>
            {mode === "high" ? <Check data-icon="inline-end" aria-hidden="true" /> : null}
          </DropdownMenuItem>
          <DropdownMenuItem
            data-intelligence-mode="pro"
            data-active={mode === "pro" ? "true" : undefined}
            onClick={() => props.onChangeMode("pro")}
          >
            <span>Pro</span>
            {mode === "pro" ? <Check data-icon="inline-end" aria-hidden="true" /> : null}
          </DropdownMenuItem>
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );

  return (
    <TooltipProvider>
      <form
        className="composer"
        data-stage={stage}
        data-layout={phone ? "phone" : "desktop"}
        data-read-mode={collapsed ? "true" : undefined}
        data-menu-open={menuOpen || mentionOpen ? "true" : undefined}
        data-editor={editorReady ? "prosemirror" : "fallback"}
        data-welcome-stage={welcomeStage ? "true" : undefined}
        onSubmit={(event) => {
          event.preventDefault();
          if (!hydrated || running || submitDisabled) {
            return;
          }
          props.onSubmit();
        }}
      >
        {menus ? <div className="composer-menus">{menus}</div> : null}
        <div ref={surfaceRef} className="composer-surface" onMouseDown={handleSurfaceMouseDown}>
          {running ? <RecorderTraceIcon className="composer-recorder" /> : null}
          {running && runningLabel ? (
            <div className="composer-eyebrow" title={runningTitle}>
              <span className="composer-run-dot" aria-hidden="true" />
              <span className="composer-eyebrow-status">{runningLabel}</span>
            </div>
          ) : null}
          <div className="composer-line">
            <ComposerAttachMenu
              disabled={!hydrated}
              onOpenResources={props.onOpenResources}
              onStartWorkflow={props.onStartWorkflow}
              onOpenNotes={props.onOpenNotes}
              onCloseAutoFocus={(event) => {
                event.preventDefault();
                current()?.focus();
              }}
            />
            <div
              className="composer-field"
              style={{ "--composer-prefix-width": `${prefixWidth}px` } as React.CSSProperties}
            >
              {value.length === 0 ? (
                <span className="composer-placeholder" aria-hidden="true">
                  {placeholder}
                </span>
              ) : null}
              <span ref={prefixRef} className="composer-prefix">
                {workflow ? (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        type="button"
                        className="composer-chip composer-chip-workflow"
                        aria-label={`Workflow: ${workflow.label}`}
                        data-testid="composer-workflow-chip"
                        onMouseDown={(event) => event.preventDefault()}
                      >
                        <Slash aria-hidden="true" />
                        <span>{workflow.label}</span>
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent
                      align="start"
                      sideOffset={8}
                      onCloseAutoFocus={(event) => {
                        event.preventDefault();
                        current()?.focus();
                      }}
                    >
                      <DropdownMenuItem onSelect={props.onChangeWorkflow}>Change workflow…</DropdownMenuItem>
                      <DropdownMenuItem onSelect={props.onRemoveWorkflow}>Remove workflow</DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                ) : null}
                {props.proIntro ? (
                  <OneTimeNotice
                    noticeId={props.proIntro.noticeId}
                    audienceId={props.proIntro.audienceId}
                    enabled={props.proIntro.enabled}
                    title={props.proIntro.title}
                    description={props.proIntro.description}
                    side="top"
                    align="start"
                    sideOffset={10}
                  >
                    {modeChip}
                  </OneTimeNotice>
                ) : (
                  modeChip
                )}
              </span>
              <Suspense fallback={<ComposerFallbackEditor ref={fallbackRef} {...editorProps} />}>
                <LazyComposerEditor ref={editorRef} {...editorProps} />
              </Suspense>
            </div>
            <div className="composer-end">
              {running ? (
                <>
                  {/* Steer, then Queue, then Stop; Stop never moves. Steer reaches
                      the RUNNING agent at its next step; Queue waits the run out. */}
                  {canSteer ? (
                    <>
                      <ComposerTooltip label="Steer this run now — ⌘↵">
                        <Button
                          size="icon"
                          type="button"
                          variant="ghost"
                          onClick={props.onSteer}
                          aria-label="Steer this run"
                          className="composer-control composer-steer"
                        >
                          <Zap size={18} />
                        </Button>
                      </ComposerTooltip>
                      <ComposerTooltip label="Queue for after this run">
                        <Button
                          size="icon"
                          type="button"
                          variant="ghost"
                          onClick={props.onQueue}
                          aria-label="Queue follow-up"
                          className="composer-control composer-queue"
                        >
                          <ArrowUp size={18} />
                        </Button>
                      </ComposerTooltip>
                    </>
                  ) : null}
                  <Button
                    size="icon"
                    type="button"
                    variant="destructive"
                    onClick={props.onStop}
                    aria-label="Stop response"
                    title="Stop response"
                    className="composer-control composer-stop"
                  >
                    <Square className="size-3.5 fill-current" />
                    <span>Stop</span>
                  </Button>
                </>
              ) : (
                <ComposerTooltip
                  disabled={submitDisabled}
                  className="app-composer-submit-tooltip"
                  label={
                    <span className="app-composer-submit-tooltip-row">
                      <span>Send prompt</span>
                      <span className="app-composer-submit-tooltip-key" aria-hidden="true">
                        ↵
                      </span>
                      <span className="sr-only">Press Enter to send. Shift+Enter starts a new line.</span>
                    </span>
                  }
                >
                  <Button
                    size="icon"
                    type="submit"
                    disabled={submitDisabled}
                    aria-label="Send prompt"
                    className="composer-control composer-send"
                  >
                    <ArrowUp size={18} />
                  </Button>
                </ComposerTooltip>
              )}
            </div>
          </div>
          {whisperVisible ? (
            <div className="composer-whisper" data-testid="composer-whisper">
              {notice ? (
                <span className="composer-whisper-summary composer-whisper-notice">
                  {notice.text}
                  {notice.action ? (
                    <>
                      {" "}
                      <button type="button" onClick={notice.action.onClick}>
                        {notice.action.label}
                      </button>
                    </>
                  ) : null}
                </span>
              ) : (
                <span className="composer-whisper-summary">{summary}</span>
              )}
              <span className="composer-whisper-keys" aria-hidden="true">
                <CornerDownLeft />
                <span>{composerKeysHint(running)}</span>
              </span>
            </div>
          ) : null}
          {extras}
          {mentionOpen && mention ? (
            <ResourceMentionPicker
              variant={phone ? "sheet" : "popover"}
              anchor={mentionAnchor === null ? undefined : { left: mentionAnchor }}
              listboxId={listboxId}
              query={mention.query}
              results={mention.results}
              loading={mention.loading}
              error={mention.error ?? null}
              activeFileId={mention.activeFileId}
              onActivate={props.onMentionActivate}
              onPick={props.onMentionPick}
              onUploadInstead={() => {
                setDismissedQuery(mention.query);
                openFilePicker();
              }}
            />
          ) : null}
        </div>
      </form>
    </TooltipProvider>
  );
});

export default Composer;
