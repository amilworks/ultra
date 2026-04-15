import { cn } from "@/lib/utils";
import {
  Steps,
  StepsBar,
  StepsContent,
  StepsItem,
  StepsTrigger,
} from "@/components/prompt-kit/steps";
import type { ProgressEvent, RunEvent } from "@/types";
import {
  CheckCircle2,
  CircleAlert,
  Loader2,
  Wrench,
  Workflow,
} from "lucide-react";
import {
  DEFAULT_THINKING_TEXT,
  getPhaseDetail,
  getPhaseLabel,
} from "@/lib/runStepCopy";

type ChatRunStepsProps = {
  runEvents: RunEvent[];
  progressEvents: ProgressEvent[];
  isStreaming: boolean;
  fallbackLabel?: string | null;
  className?: string;
};

type StepStatus = "running" | "completed" | "failed";
type StepKind = "phase" | "tool";

type ChatStepItem = {
  id: string;
  kind: StepKind;
  label: string;
  detail: string | null;
  status: StepStatus;
};

const TOOL_LABELS: Record<string, string> = {
  bisque_advanced_search: "Advanced BisQue search",
  bisque_download_resource: "Download BisQue resource",
  bisque_find_assets: "Find BisQue assets",
  create_bisque_dataset: "Create BisQue dataset",
  estimate_depth_depthpro: "DepthPro estimation",
  resource_lookup: "Resolve uploaded resources",
  sam2_prompt_image: "SAM2 prompting",
  search_bisque_resources: "Search BisQue resources",
  segment_image_sam2: "MedSAM2 segmentation",
  segment_image_sam3: "SAM3 segmentation",
  yolo_detect: "YOLO detection",
};

const toRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
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
      if (lower === "sam3") return "SAM3";
      if (lower === "sam2") return "SAM2";
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

const buildStepItems = (
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

  return items.slice(0, 12);
};

const renderStepIcon = (item: ChatStepItem) => {
  if (item.status === "running") {
    return <Loader2 className="size-4 animate-spin" />;
  }
  if (item.status === "failed") {
    return <CircleAlert className="size-4" />;
  }
  if (item.kind === "tool") {
    return <Wrench className="size-4" />;
  }
  return <CheckCircle2 className="size-4" />;
};

export function ChatRunSteps({
  runEvents,
  progressEvents,
  isStreaming,
  fallbackLabel,
  className,
}: ChatRunStepsProps) {
  if (!isStreaming) {
    return null;
  }
  const stepItems = buildStepItems(runEvents, progressEvents);
  if (stepItems.length === 0) {
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
  const triggerText = String(
    fallbackLabel || activeItem?.label || lastItem?.label || DEFAULT_THINKING_TEXT
  ).trim();

  return (
    <Steps
      defaultOpen
      className={cn("chat-run-steps w-full", className)}
      data-testid="chat-run-steps"
    >
      <StepsTrigger
        className="chat-run-steps-trigger text-left"
        leftIcon={
          isStreaming ? (
            <Loader2 className="size-4 animate-spin" />
          ) : hasFailure ? (
            <CircleAlert className="size-4" />
          ) : (
            <Workflow className="size-4" />
          )
        }
      >
        {`Steps: ${triggerText}`}
      </StepsTrigger>
      <StepsContent
        className="chat-run-steps-content"
        bar={
          <StepsBar
            className={cn(
              "ml-1.5 mr-2",
              hasFailure
                ? "bg-destructive/35"
                : isStreaming
                  ? "bg-primary/35"
                  : "bg-muted"
            )}
          />
        }
      >
        <div className="space-y-2">
          {stepItems.map((item) => (
            <StepsItem key={item.id} className="chat-run-step-item">
              <div className="flex items-start gap-2">
                <span
                  className={cn(
                    "mt-0.5 shrink-0",
                    item.status === "failed"
                      ? "text-destructive"
                      : item.status === "running"
                        ? "text-primary"
                        : item.kind === "tool"
                          ? "text-emerald-600 dark:text-emerald-400"
                          : "text-foreground"
                  )}
                >
                  {renderStepIcon(item)}
                </span>
                <div className="min-w-0">
                  <div
                    className={cn(
                      "text-sm leading-5",
                      item.status === "failed"
                        ? "text-destructive"
                        : item.status === "running"
                          ? "text-foreground"
                          : "text-foreground/90"
                    )}
                  >
                    {item.label}
                  </div>
                  {item.detail ? (
                    <p className="text-muted-foreground mt-0.5 text-xs leading-5">
                      {item.detail}
                    </p>
                  ) : null}
                </div>
              </div>
            </StepsItem>
          ))}
        </div>
      </StepsContent>
    </Steps>
  );
}
