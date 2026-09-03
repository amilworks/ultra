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
import { ArrowUp, Check, Slash, Square, Zap } from "lucide-react";

import { ResourceMentionPicker } from "@/components/chat/ResourceMentionPicker";
import { RecorderTraceIcon } from "@/components/icons/MeridianIcons";
import { useFileUploadContext } from "@/components/prompt-kit/file-upload";
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
import { cn } from "@/lib/utils";
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
  composerPlaceholder,
  composerSummary,
  deriveComposerStage,
} from "./composerModel";

/* The composer: one bar.

   Its geometry is two heights and nothing else. At rest it is the bar alone,
   a sunk well: attach, the hint, the mode tag, send. While composing it rises
   into a card whose text block sits ABOVE that same bar — the bar never
   changes, so nothing in it ever moves relative to the text. Every control,
   tag and line of type centres on the bar's axis; the text block has its own
   padding and never shares a row with a control, so there is nothing to
   misalign. Files live in the text as tokens; the workflow and the mode live
   in the bar as mono tags; the bar's middle carries the brief's summary, a
   notice, or the run's eyebrow.

   States are explicit (see composerModel) and stamped on the form as data
   attributes; the stylesheet only draws them. */

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
  /** Scrolled away from the bottom of a long answer: the bar alone, with a hint. */
  readMode: boolean;
  phone: boolean;
  welcomeStage: boolean;
  /** A slash menu or the library picker is open above the bar. */
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

  /** Slash workflows and the library picker, rendered by the app above the bar. */
  menus?: ReactNode;
  /** Notes chips and pending-upload previews, rendered between the text and the bar. */
  extras?: ReactNode;

  runningLabel?: ReactNode;
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
  target.closest(
    "button, a, [role='menu'], [role='listbox'], .composer-mention-picker, .composer-menus"
  ) !== null;

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
  const cardRef = useRef<HTMLDivElement | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;

  const [focused, setFocused] = useState(false);
  const focusedRef = useRef(false);
  const [editorReady, setEditorReady] = useState(false);
  const [dismissedQuery, setDismissedQuery] = useState<string | null>(null);
  const [mentionAnchor, setMentionAnchor] = useState<number | null>(null);
  const listboxId = useId();
  const { openFilePicker } = useFileUploadContext();

  const current = useCallback(
    (): ComposerHandle | null => editorRef.current ?? fallbackRef.current,
    []
  );

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
  const mentionOpen = mention !== null && dismissedQuery !== mention.query;
  const stage = deriveComposerStage({
    running,
    focused,
    hasText,
    hasTokens: tokens.length > 0,
    hasFiles,
    hasWorkflow: workflow !== null,
    menuOpen: menuOpen || mentionOpen,
    welcomeStage,
  });
  const collapsed = readMode && !running && !focused;
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
     relative to the card, clamped so the popover stays inside it. */
  useLayoutEffect(() => {
    if (!mentionOpen || phone) {
      setMentionAnchor(null);
      return;
    }
    const rect = current()?.mentionRect();
    const card = cardRef.current?.getBoundingClientRect();
    if (!rect || !card) {
      setMentionAnchor(null);
      return;
    }
    setMentionAnchor(clampMentionAnchor(rect.left - card.left, card.width));
  }, [mentionOpen, phone, mention?.query, current]);

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
        if (
          (event.key === "ArrowDown" || event.key === "ArrowUp") &&
          activeMention.results.length > 0
        ) {
          event.preventDefault();
          const ids = activeMention.results.map((resource) => resource.file_id);
          const index = ids.indexOf(activeMention.activeFileId ?? "");
          const step = event.key === "ArrowDown" ? 1 : -1;
          const next =
            index < 0 ? (step > 0 ? 0 : ids.length - 1) : (index + step + ids.length) % ids.length;
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
              activeMention.results.find(
                (resource) => resource.file_id === activeMention.activeFileId
              ) ?? activeMention.results[0];
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

  /* The whole card is the field: a click anywhere that is not a control
     lands the caret at the end. Clicks inside the editor place it themselves. */
  const handleCardMouseDown = useCallback(
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
    mentionOpen && mention?.activeFileId
      ? resourceMentionOptionId(listboxId, mention.activeFileId)
      : undefined;

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

  /* The bar's middle. Priority: a notice that blocks the send, the run's
     eyebrow, the read-mode instruction, the resting hint (the field's own
     voice), then the brief's summary once there is something to summarise. */
  let status: ReactNode = null;
  let statusClass = "composer-status";
  if (hydrated && notice) {
    statusClass += " composer-status-notice";
    status = (
      <>
        {notice.text}
        {notice.action ? (
          <>
            {" "}
            <button type="button" onClick={notice.action.onClick}>
              {notice.action.label}
            </button>
          </>
        ) : null}
      </>
    );
  } else if (running) {
    statusClass += " composer-eyebrow";
    status = (
      <>
        <span className="composer-run-dot" aria-hidden="true" />
        <span className="composer-status-text">{runningLabel}</span>
      </>
    );
  } else if (collapsed) {
    status = placeholder;
  } else if (stage === "rest") {
    statusClass += " composer-status-hint";
    status = placeholder;
  } else if (hasText || tokens.length > 0 || workflow !== null) {
    status = summary;
  }

  const modeLabel = mode === "pro" ? "Pro" : "High";
  const modeTag = (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className={cn("composer-tag composer-tag-mode", mode === "pro" && "composer-tag-strong")}
          aria-label={`Intelligence: ${modeLabel}`}
          data-testid="composer-mode-tag"
          disabled={!hydrated}
          onMouseDown={(event) => event.preventDefault()}
        >
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

  const picker =
    mentionOpen && mention ? (
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
    ) : null;

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
        <div ref={cardRef} className="composer-card" onMouseDown={handleCardMouseDown}>
          {running ? <RecorderTraceIcon className="composer-recorder" /> : null}
          <div className="composer-text">
            {value.length === 0 ? (
              <span className="composer-placeholder" aria-hidden="true">
                {placeholder}
              </span>
            ) : null}
            <Suspense fallback={<ComposerFallbackEditor ref={fallbackRef} {...editorProps} />}>
              <LazyComposerEditor ref={editorRef} {...editorProps} />
            </Suspense>
          </div>
          {extras}
          {phone ? picker : null}
          <div className="composer-bar">
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
            {workflow ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    type="button"
                    className="composer-tag composer-tag-workflow"
                    aria-label={`Workflow: ${workflow.label}`}
                    data-testid="composer-workflow-tag"
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
                {modeTag}
              </OneTimeNotice>
            ) : (
              modeTag
            )}
            <div className={statusClass} title={running ? runningTitle : undefined}>
              {status}
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
                      <span className="sr-only">
                        Press Enter to send. Shift+Enter starts a new line.
                      </span>
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
          {phone ? null : picker}
        </div>
      </form>
    </TooltipProvider>
  );
});

export default Composer;
