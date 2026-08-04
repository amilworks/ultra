import { useMemo } from "react";

import { cn } from "@/lib/utils";
import { toRecord } from "@/lib/coerce";
import {
  Steps,
  StepsBar,
  StepsContent,
  StepsItem,
  StepsTrigger,
} from "@/components/prompt-kit/steps";
import { TextShimmer } from "@/components/prompt-kit/text-shimmer";
import { ThinkingConstellation } from "@/components/chat/ThinkingConstellation";
import type { ProgressEvent, RunEvent } from "@/types";
import { Check, Circle, CircleAlert, CircleDot, Loader2 } from "lucide-react";
import {
  DEFAULT_THINKING_TEXT,
  getPhaseDetail,
  getPhaseLabel,
} from "@/lib/runStepCopy";

type ChatRunStepsProps = {
  runEvents: RunEvent[];
  progressEvents: ProgressEvent[];
  isStreaming: boolean;
  /** Calm phase-level status (from runStepCopy) shown when no tool step is active. */
  statusText?: string | null;
  onStop?: () => void;
  stopLabel?: string;
  className?: string;
};

type StepStatus = "running" | "completed" | "failed";
type StepKind = "phase" | "tool";

type PlanTodoStatus = "pending" | "in_progress" | "completed";

type PlanTodo = {
  content: string;
  status: PlanTodoStatus;
};

type ChatStepItem = {
  id: string;
  kind: StepKind;
  label: string;
  detail: string | null;
  status: StepStatus;
  /** Present only on the "plan" step: the latest write_todos list. */
  todos?: PlanTodo[];
};

// Calm, human phrases in the present tense — never raw tool identifiers.
const TOOL_LABELS: Record<string, string> = {
  artifact_manifest: "Reviewing prior outputs",
  bisque_advanced_search: "Searching your BisQue library",
  bisque_download_resource: "Fetching a BisQue resource",
  bisque_find_assets: "Finding BisQue assets",
  bisque_search_resources: "Searching your BisQue library",
  create_bisque_dataset: "Creating a BisQue dataset",
  edit_file: "Editing a file",
  estimate_depth_depthpro: "Estimating depth",
  execute: "Running code",
  glob: "Looking through files",
  grep: "Searching the files",
  ls: "Looking through files",
  read_file: "Reading a file",
  resource_lookup: "Resolving your resources",
  search_bisque_resources: "Searching your BisQue library",
  search_resources: "Searching your resources",
  stage_git_repo_for_analysis: "Cloning a repository",
  stage_resource_for_analysis: "Staging a resource",
  stage_uploaded_files_for_analysis: "Preparing your files",
  write_file: "Writing a file",
  write_todos: "Planning the work",
  yolo_detect: "Detecting objects",
};

// The write_todos input is the FULL todo list each call (replace semantics), so
// the newest parsed list is always the current plan.
const parsePlanTodos = (value: unknown): PlanTodo[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const todos: PlanTodo[] = [];
  for (const entry of value) {
    const record = toRecord(entry);
    const content = String(record?.content ?? "").trim();
    if (!content) {
      continue;
    }
    const raw = String(record?.status ?? "").trim().toLowerCase();
    const status: PlanTodoStatus =
      raw === "completed" ? "completed" : raw === "in_progress" ? "in_progress" : "pending";
    todos.push({ content, status });
  }
  return todos;
};

const toStepStatus = (value: unknown): StepStatus => {
  const normalized = String(value || "").trim().toLowerCase();
  if (
    normalized === "failed" ||
    normalized === "error" ||
    normalized === "canceled" ||
    normalized === "cancelled"
  ) {
    return "failed";
  }
  if (
    normalized === "running" ||
    normalized === "started" ||
    normalized === "queued" ||
    normalized === "progress"
  ) {
    return "running";
  }
  return "completed";
};

const titleCaseWords = (value: string): string =>
  value
    .split(/[_\-\s]+/g)
    .filter(Boolean)
    .map((token) => {
      const lower = token.toLowerCase();
      if (lower === "yolo") return "YOLO";
      if (lower === "bisque") return "BisQue";
      return lower[0] ? lower[0].toUpperCase() + lower.slice(1) : lower;
    })
    .join(" ");

const formatToolLabel = (toolName: string): string => {
  const normalized = toolName.trim().toLowerCase();
  if (!normalized) {
    return "Run tool";
  }
  return TOOL_LABELS[normalized] ?? titleCaseWords(normalized);
};

const formatPhaseLabel = (phase: string): string => {
  const normalized = phase.trim().toLowerCase();
  if (!normalized) {
    return "Progress update";
  }
  return getPhaseLabel(normalized) ?? "Progress update";
};

const cleanDetail = (value: unknown): string | null => {
  const normalized = String(value || "").trim();
  return normalized.length > 0 ? normalized : null;
};

const statusFromV2ToolEvent = (eventType: string, payload: Record<string, unknown>): StepStatus => {
  const status = String(payload.status || "").trim();
  if (status) {
    return toStepStatus(status);
  }
  if (eventType.endsWith(".started")) {
    return "running";
  }
  if (eventType.endsWith(".failed") || eventType.endsWith(".error")) {
    return "failed";
  }
  return "completed";
};

// Exported for golden testing: the canonical run-event → step grouping. It is a PURE, deterministic
// function of the ordered (runEvents, progressEvents) — the same inputs always coalesce into the same
// bounded set of foldable steps (e.g. all reasoning deltas → one "Thinking" step; a tool's
// started+completed → one tool step). This is the RunStep model; keeping it deterministic is what lets
// the trace render identically on reconnect / across replicas.
export const buildStepItems = (
  runEvents: RunEvent[],
  progressEvents: ProgressEvent[]
): ChatStepItem[] => {
  const items: ChatStepItem[] = [];
  const byKey = new Map<string, number>();

  const upsertStep = (key: string, nextItem: ChatStepItem) => {
    const existingIndex = byKey.get(key);
    if (existingIndex === undefined) {
      byKey.set(key, items.length);
      items.push(nextItem);
      return;
    }
    items[existingIndex] = {
      ...items[existingIndex],
      ...nextItem,
      detail: nextItem.detail ?? items[existingIndex].detail,
      // tool_call.completed events carry no input, so a plan update without a
      // parsed list must not wipe the checklist rendered from .started.
      todos: nextItem.todos ?? items[existingIndex].todos,
    };
  };

  runEvents.forEach((event, index) => {
    const payload = toRecord(event.payload);
    if (!payload) {
      return;
    }
    const nestedPayload = toRecord(payload.payload);
    const eventType = String(event.event_type || "").trim().toLowerCase();
    const detail = cleanDetail(payload.message);
    if (eventType === "trace.reasoning.delta") {
      upsertStep("reasoning", {
        id: "reasoning",
        kind: "phase",
        label: "Thinking",
        detail: cleanDetail(payload.text),
        status: toStepStatus(payload.status ?? "running"),
      });
      return;
    }
    if (eventType.startsWith("tool_call.")) {
      const toolName = String(payload.tool_name ?? payload.tool ?? "").trim();
      if (!toolName) {
        return;
      }
      if (toolName.toLowerCase() === "write_todos") {
        // One durable "plan" step for the whole run: write_todos replaces the
        // full list each call, so the latest event's todos are the current plan.
        const todos = parsePlanTodos(toRecord(payload.input)?.todos);
        const done = todos.filter((todo) => todo.status === "completed").length;
        upsertStep("plan", {
          id: "plan",
          kind: "phase",
          label: "Planning the work",
          detail: todos.length > 0 ? `${done} of ${todos.length} done` : null,
          status: statusFromV2ToolEvent(eventType, payload),
          todos: todos.length > 0 ? todos : undefined,
        });
        return;
      }
      const invocationId = String(
        payload.tool_call_id ?? payload.call_id ?? payload.event_id ?? `${toolName}-${index}`
      ).trim();
      upsertStep(`tool:${invocationId}`, {
        id: `tool:${invocationId}`,
        kind: "tool",
        label: formatToolLabel(toolName),
        detail:
          detail ??
          cleanDetail(payload.text) ??
          cleanDetail(payload.output_tail) ??
          cleanDetail(payload.command_preview) ??
          cleanDetail(payload.command),
        status: statusFromV2ToolEvent(eventType, payload),
      });
      return;
    }
    if (
      eventType === "memory.retrieved" ||
      eventType === "memory.updated" ||
      eventType === "knowledge.retrieved" ||
      eventType === "learning.promoted" ||
      eventType === "learning.skipped"
    ) {
      const phase =
        eventType.startsWith("memory.")
          ? "memory"
          : eventType.startsWith("knowledge.")
            ? "knowledge"
            : "learning";
      upsertStep(`graph:${phase}:${eventType}`, {
        id: `graph:${phase}:${eventType}`,
        kind: "phase",
        label: formatPhaseLabel(phase),
        detail: getPhaseDetail(phase),
        status: toStepStatus(payload.status),
      });
      return;
    }
    if (eventType === "tool_event" || eventType === "pro_mode.tool_requested" || eventType === "pro_mode.tool_completed") {
      const toolName = String(
        nestedPayload?.tool ?? payload.tool ?? payload.node ?? ""
      ).trim();
      if (!toolName) {
        return;
      }
      const invocationId = String(
        nestedPayload?.invocation_id ?? payload.invocation_id ?? `${toolName}-${index}`
      ).trim();
      upsertStep(`tool:${invocationId}`, {
        id: `tool:${invocationId}`,
        kind: "tool",
        label: formatToolLabel(toolName),
        detail,
        status: toStepStatus(payload.status),
      });
      return;
    }
    if (eventType === "pro_mode.role_message") {
      return;
    }
    if (eventType !== "graph_event" && eventType !== "pro_mode.phase_started" && eventType !== "pro_mode.phase_completed" && eventType !== "pro_mode.convergence_updated" && eventType !== "pro_mode.verifier_result") {
      return;
    }
    const phase = String(payload.phase || "").trim();
    if (!phase) {
      return;
    }
    const domainToken = String(
      payload.domain_id ?? nestedPayload?.domain_id ?? payload.node ?? ""
    ).trim();
    const key =
      phase.toLowerCase() === "solve" && domainToken
        ? `graph:${phase}:${domainToken}`
        : `graph:${phase}`;
    upsertStep(key, {
      id: key,
      kind: "phase",
      label: formatPhaseLabel(phase),
      detail: getPhaseDetail(phase),
      status: toStepStatus(payload.status),
    });
  });

  if (items.length === 0) {
    progressEvents.forEach((event, index) => {
      const toolName = String(event.tool || "").trim();
      if (!toolName) {
        return;
      }
      const key = `progress:${toolName}:${index}`;
      upsertStep(key, {
        id: key,
        kind: "tool",
        label: formatToolLabel(toolName),
        detail: cleanDetail(event.message),
        status: toStepStatus(event.event),
      });
    });
  }

  // Collapse repeated identical labels (e.g. two "Looking through files") into a
  // single calm line, keeping first-seen order and the most recent status.
  const deduped: ChatStepItem[] = [];
  const indexByLabel = new Map<string, number>();
  for (const item of items) {
    const existingIndex = indexByLabel.get(item.label);
    if (existingIndex === undefined) {
      indexByLabel.set(item.label, deduped.length);
      deduped.push(item);
    } else {
      deduped[existingIndex] = { ...deduped[existingIndex], status: item.status };
    }
  }

  return deduped.slice(0, 12);
};

const renderPlanTodoIcon = (status: PlanTodoStatus) => {
  if (status === "completed") {
    return <Check className="size-3" />;
  }
  if (status === "in_progress") {
    return <CircleDot className="size-3" />;
  }
  return <Circle className="size-3" />;
};

const renderStepIcon = (item: ChatStepItem) => {
  if (item.status === "running") {
    return <Loader2 className="size-3.5 animate-spin" />;
  }
  if (item.status === "failed") {
    return <CircleAlert className="size-3.5" />;
  }
  return <Check className="size-3.5" />;
};

function StopAffordance({ onStop, stopLabel }: { onStop: () => void; stopLabel: string }) {
  return (
    <button
      type="button"
      onClick={(event) => {
        // The trigger wraps this row; don't toggle the steps when stopping.
        event.stopPropagation();
        onStop();
      }}
      className="text-muted-foreground hover:text-foreground border-muted-foreground/50 hover:border-foreground shrink-0 border-b border-dotted text-sm transition-colors"
    >
      {stopLabel}
    </button>
  );
}

export function ChatRunSteps({
  runEvents,
  progressEvents,
  isStreaming,
  statusText,
  onStop,
  stopLabel = "Stop",
  className,
}: ChatRunStepsProps) {
  // Memoize the step derivation so it only recomputes when the event arrays actually change.
  // Their references change on STRUCTURAL events only — token deltas never touch
  // runEvents/progressEvents — so this skips the O(events) walk on every streaming re-render.
  // (Hook stays above the early return to satisfy the rules-of-hooks ordering.)
  const stepItems = useMemo(
    () => buildStepItems(runEvents, progressEvents),
    [runEvents, progressEvents]
  );
  if (!isStreaming) {
    return null;
  }

  let activeItem: ChatStepItem | null = null;
  for (let index = stepItems.length - 1; index >= 0; index -= 1) {
    if (stepItems[index]?.status === "running") {
      activeItem = stepItems[index];
      break;
    }
  }
  const lastItem = stepItems[stepItems.length - 1] ?? null;
  const hasFailure = stepItems.some((item) => item.status === "failed");
  const liveLabel = String(
    activeItem?.label || statusText || lastItem?.label || DEFAULT_THINKING_TEXT
  ).trim();
  const hasSteps = stepItems.length > 0;

  // The constellation is the activity pulse; the shimmer is the words. One
  // gesture together, in both layouts below.
  const shimmer = (
    <span className="flex min-w-0 items-center gap-2">
      <ThinkingConstellation className="shrink-0" />
      <TextShimmer className="min-w-0 truncate font-medium">{liveLabel}</TextShimmer>
    </span>
  );

  // No steps yet: a single calm shimmer line + Stop, no chevron.
  if (!hasSteps) {
    return (
      <div className={cn("chat-run-steps flex w-full items-center justify-between gap-3", className)}>
        {shimmer}
        {onStop ? <StopAffordance onStop={onStop} stopLabel={stopLabel} /> : null}
      </div>
    );
  }

  // Steps exist: same calm line, now an expandable disclosure (collapsed by default).
  return (
    <Steps
      defaultOpen={false}
      className={cn("chat-run-steps w-full", className)}
      data-testid="chat-run-steps"
    >
      <div className="flex w-full items-center justify-between gap-3">
        <StepsTrigger className="w-auto min-w-0 flex-1">{shimmer}</StepsTrigger>
        {onStop ? <StopAffordance onStop={onStop} stopLabel={stopLabel} /> : null}
      </div>
      <StepsContent
        className="chat-run-steps-content"
        bar={
          <StepsBar
            className={cn("ml-1.5 mr-2", hasFailure ? "bg-destructive/30" : "bg-muted")}
          />
        }
      >
        <div className="space-y-1.5">
          {stepItems.map((item) => (
            <StepsItem key={item.id} className="chat-run-step-item">
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    "shrink-0",
                    item.status === "failed"
                      ? "text-destructive/70"
                      : item.status === "running"
                        ? "text-muted-foreground"
                        : "text-muted-foreground/60"
                  )}
                >
                  {renderStepIcon(item)}
                </span>
                <span
                  className={cn(
                    "min-w-0 truncate text-sm leading-5",
                    item.status === "running" ? "text-foreground/80" : "text-muted-foreground"
                  )}
                >
                  {item.label}
                </span>
              </div>
              {item.id === "reasoning" && item.detail ? (
                <div className="chat-run-step-reasoning">{item.detail}</div>
              ) : null}
              {item.id === "plan" && item.todos && item.todos.length > 0 ? (
                <div className="chat-run-step-plan">
                  {item.todos.map((todo) => (
                    <div key={todo.content} className="flex items-start gap-2">
                      <span
                        className={cn(
                          "mt-1 shrink-0",
                          todo.status === "in_progress"
                            ? "text-foreground/70"
                            : "text-muted-foreground/50"
                        )}
                      >
                        {renderPlanTodoIcon(todo.status)}
                      </span>
                      <span
                        className={cn(
                          "min-w-0 text-sm leading-5",
                          todo.status === "in_progress"
                            ? "text-foreground/80"
                            : todo.status === "completed"
                              ? "text-muted-foreground/60"
                              : "text-muted-foreground"
                        )}
                      >
                        {todo.content}
                      </span>
                    </div>
                  ))}
                </div>
              ) : null}
            </StepsItem>
          ))}
        </div>
      </StepsContent>
    </Steps>
  );
}
