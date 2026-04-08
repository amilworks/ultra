import { useEffect, useMemo, useRef } from "react";
import {
  Check,
  FolderSearch,
  ImageIcon,
  Link2,
  Loader2,
  Sparkles,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { cn } from "@/lib/utils";
import { formatBytes } from "@/lib/format";
import type { ResourceRecord } from "@/types";

import {
  COMPOSER_WORKFLOW_GROUP_ORDER,
  filterComposerWorkflows,
  type ComposerWorkflowDefinition,
  type ComposerWorkflowId,
  type ComposerWorkflowPresetState,
} from "./composer-workflows";

type ComposerSlashMenuProps = {
  mode: "workflow" | "resource_picker";
  workflowQuery?: string;
  activeWorkflowId?: ComposerWorkflowId | null;
  onSelectWorkflow?: (workflow: ComposerWorkflowDefinition) => void;
  preset?: ComposerWorkflowPresetState | null;
  resourceQuery?: string;
  onResourceQueryChange?: (value: string) => void;
  resources?: ResourceRecord[];
  resourcesLoading?: boolean;
  resourcesError?: string | null;
  activeResourceId?: string | null;
  selectedResourceIds?: Set<string>;
  onResourceInputKeyDown?: React.KeyboardEventHandler<HTMLInputElement>;
  onToggleResource?: (resource: ResourceRecord) => void;
  onConfirmResources?: () => void;
  onCancelResourcePicker?: () => void;
};

const resourceKindLabel = (kind: string): string => {
  const normalized = String(kind || "").trim().toLowerCase();
  if (!normalized) {
    return "resource";
  }
  return normalized;
};

const formatResourceDate = (value: string): string => {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) {
    return "Unknown date";
  }
  return new Date(timestamp).toLocaleDateString();
};

const ACTIVE_COMMAND_ITEM_SELECTOR = '[data-composer-active="true"]';
const COMMAND_ITEM_SCROLL_PADDING = 12;

const scrollActiveCommandItemIntoView = (container: HTMLDivElement | null): void => {
  if (!container) {
    return;
  }
  const activeItem = container.querySelector<HTMLElement>(ACTIVE_COMMAND_ITEM_SELECTOR);
  if (!activeItem) {
    return;
  }

  const containerRect = container.getBoundingClientRect();
  const activeItemRect = activeItem.getBoundingClientRect();
  const containerTop = container.scrollTop;
  const containerBottom = containerTop + container.clientHeight;
  const itemTop = activeItemRect.top - containerRect.top + container.scrollTop;
  const itemBottom = itemTop + activeItemRect.height;

  if (itemTop < containerTop + COMMAND_ITEM_SCROLL_PADDING) {
    container.scrollTop = Math.max(itemTop - COMMAND_ITEM_SCROLL_PADDING, 0);
    return;
  }
  if (itemBottom > containerBottom - COMMAND_ITEM_SCROLL_PADDING) {
    container.scrollTop =
      itemBottom - container.clientHeight + COMMAND_ITEM_SCROLL_PADDING;
  }
};

export function ComposerSlashMenu({
  mode,
  workflowQuery = "",
  activeWorkflowId = null,
  onSelectWorkflow,
  preset = null,
  resourceQuery = "",
  onResourceQueryChange,
  resources = [],
  resourcesLoading = false,
  resourcesError = null,
  activeResourceId = null,
  selectedResourceIds = new Set<string>(),
  onResourceInputKeyDown,
  onToggleResource,
  onConfirmResources,
  onCancelResourcePicker,
}: ComposerSlashMenuProps) {
  const workflowListRef = useRef<HTMLDivElement | null>(null);
  const resourceListRef = useRef<HTMLDivElement | null>(null);

  const filteredWorkflows = useMemo(
    () => filterComposerWorkflows(workflowQuery),
    [workflowQuery]
  );

  const groupedWorkflows = useMemo(() => {
    const grouped = new Map<
      ComposerWorkflowDefinition["category"],
      ComposerWorkflowDefinition[]
    >();
    filteredWorkflows.forEach((workflow) => {
      const existing = grouped.get(workflow.category) ?? [];
      existing.push(workflow);
      grouped.set(workflow.category, existing);
    });
    return COMPOSER_WORKFLOW_GROUP_ORDER.map((category) => ({
      category,
      items: grouped.get(category) ?? [],
    })).filter((group) => group.items.length > 0);
  }, [filteredWorkflows]);

  useEffect(() => {
    if (mode !== "workflow" || !activeWorkflowId) {
      return;
    }
    scrollActiveCommandItemIntoView(workflowListRef.current);
  }, [mode, activeWorkflowId, groupedWorkflows]);

  useEffect(() => {
    if (mode !== "resource_picker" || !activeResourceId) {
      return;
    }
    scrollActiveCommandItemIntoView(resourceListRef.current);
  }, [mode, activeResourceId, resources]);

  if (mode === "workflow") {
    return (
      <div
        className="absolute right-0 bottom-[calc(100%+0.75rem)] left-0 z-30"
        data-testid="composer-slash-menu"
      >
        <div className="overflow-hidden rounded-[1.4rem] border border-border/70 bg-popover/95 shadow-2xl backdrop-blur">
          <div className="flex items-center justify-between gap-3 border-b border-border/60 px-4 py-3">
            <div className="min-w-0">
              <p className="text-foreground flex items-center gap-2 text-sm font-medium">
                <Sparkles className="size-4 text-primary" />
                Slash workflows
              </p>
              <p className="text-muted-foreground text-xs">
                Choose a structured workflow for this turn.
              </p>
            </div>
            <Badge variant="outline" className="shrink-0 rounded-full px-2 py-0.5 text-[11px]">
              type `/`
            </Badge>
          </div>
          <Command className="h-auto bg-transparent" shouldFilter={false}>
            <CommandList ref={workflowListRef} className="max-h-[360px] px-2 py-2">
              <CommandEmpty className="py-5 text-sm text-muted-foreground">
                No workflows matched that slash query.
              </CommandEmpty>
              {groupedWorkflows.map((group) => (
                <CommandGroup key={group.category} heading={group.category}>
                  {group.items.map((workflow) => {
                    const Icon = workflow.icon;
                    const active = workflow.id === activeWorkflowId;
                    return (
                      <CommandItem
                        key={workflow.id}
                        value={workflow.id}
                        data-testid={`composer-workflow-${workflow.id}`}
                        data-composer-active={active ? "true" : undefined}
                        className={cn(
                          "items-center gap-3 rounded-xl px-3 py-3",
                          active && "bg-accent text-accent-foreground"
                        )}
                        onMouseDown={(event) => event.preventDefault()}
                        onSelect={() => onSelectWorkflow?.(workflow)}
                      >
                        <div
                          className={cn(
                            "flex size-9 shrink-0 items-center justify-center rounded-full border",
                            active
                              ? "border-primary/35 bg-primary/12 text-primary"
                              : "border-border/70 bg-background text-muted-foreground"
                          )}
                        >
                          <Icon className="size-4" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium">{workflow.label}</p>
                          <p className="text-muted-foreground line-clamp-2 text-xs">
                            {workflow.description}
                          </p>
                        </div>
                        {active ? <Check className="size-4 shrink-0 text-primary" /> : null}
                      </CommandItem>
                    );
                  })}
                </CommandGroup>
              ))}
            </CommandList>
          </Command>
        </div>
      </div>
    );
  }

  const selectedCount = selectedResourceIds.size;
  const resourceTitle = preset?.id === "find_resource" ? "Find resources" : `Choose resources for ${preset?.label ?? "this workflow"}`;
  const resourceDescription =
    preset?.id === "find_resource"
      ? "Search your resource catalog and stage files into this chat."
      : "Select one or more resources to stage before sending the workflow-backed prompt.";

  return (
    <div
      className="absolute right-0 bottom-[calc(100%+0.75rem)] left-0 z-30"
      data-testid="composer-resource-picker"
    >
      <div className="overflow-hidden rounded-[1.4rem] border border-border/70 bg-popover/95 shadow-2xl backdrop-blur">
        <div className="flex items-start justify-between gap-3 border-b border-border/60 px-4 py-3">
          <div className="min-w-0">
            <p className="text-foreground flex items-center gap-2 text-sm font-medium">
              <FolderSearch className="size-4 text-primary" />
              {resourceTitle}
            </p>
            <p className="text-muted-foreground text-xs">{resourceDescription}</p>
          </div>
          {selectedCount > 0 ? (
            <Badge variant="secondary" className="shrink-0 rounded-full px-2.5 py-0.5 text-[11px]">
              {selectedCount} selected
            </Badge>
          ) : null}
        </div>
        <Command className="h-auto bg-transparent" shouldFilter={false}>
          <CommandInput
            autoFocus
            value={resourceQuery}
            onValueChange={onResourceQueryChange}
            onKeyDown={onResourceInputKeyDown}
            placeholder="Search files, BisQue IDs, or URLs"
            aria-label="Find resources"
          />
          <CommandList ref={resourceListRef} className="max-h-[340px] px-2 py-2">
            {resourcesError ? (
              <div className="px-3 py-4 text-sm text-destructive">{resourcesError}</div>
            ) : null}
            {resourcesLoading ? (
              <div className="flex items-center gap-2 px-3 py-4 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                Loading resources…
              </div>
            ) : null}
            {!resourcesLoading && !resourcesError ? (
              <CommandEmpty className="py-5 text-sm text-muted-foreground">
                No resources matched that search.
              </CommandEmpty>
            ) : null}
            {!resourcesLoading && !resourcesError ? (
              <CommandGroup heading="Resources">
                {resources.map((resource) => {
                  const selected = selectedResourceIds.has(resource.file_id);
                  const active = resource.file_id === activeResourceId;
                  return (
                    <CommandItem
                      key={resource.file_id}
                      value={`${resource.original_name} ${resource.file_id} ${resource.source_uri ?? ""}`}
                      data-composer-active={active ? "true" : undefined}
                      className={cn(
                        "items-start gap-3 rounded-xl px-3 py-3",
                        active && "bg-accent/70 text-accent-foreground",
                        selected && "bg-accent text-accent-foreground"
                      )}
                      onMouseDown={(event) => event.preventDefault()}
                      onSelect={() => onToggleResource?.(resource)}
                    >
                      <div
                        aria-hidden="true"
                        className={cn(
                          "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-md border transition-colors",
                          selected
                            ? "border-primary bg-primary text-primary-foreground"
                            : "border-border/70 bg-background text-transparent"
                        )}
                      >
                        <Check className="size-3.5" />
                      </div>
                      <div className="flex min-w-0 flex-1 items-start gap-3">
                        <div className="flex size-9 shrink-0 items-center justify-center rounded-full border border-border/70 bg-background text-muted-foreground">
                          <ImageIcon className="size-4" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium">{resource.original_name}</p>
                          <p className="text-muted-foreground text-xs">
                            {formatBytes(resource.size_bytes)} • {formatResourceDate(resource.created_at)}
                          </p>
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            <Badge variant="outline" className="rounded-full px-2 py-0 text-[10px]">
                              {resource.source_type}
                            </Badge>
                            <Badge variant="outline" className="rounded-full px-2 py-0 text-[10px]">
                              {resourceKindLabel(resource.resource_kind)}
                            </Badge>
                          </div>
                          {resource.source_uri ? (
                            <p className="text-muted-foreground mt-2 flex items-center gap-1 text-[11px]">
                              <Link2 className="size-3" />
                              <span className="truncate">{resource.source_uri}</span>
                            </p>
                          ) : null}
                        </div>
                      </div>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            ) : null}
          </CommandList>
        </Command>
        <div className="flex items-center justify-between gap-3 border-t border-border/60 px-4 py-3">
          <p className="text-muted-foreground text-xs">
            {preset?.id === "find_resource"
              ? "Selected resources will be staged into the current chat."
              : "Stage resources first, then send the workflow-backed prompt."}
          </p>
          <div className="flex items-center gap-2">
            <Button type="button" variant="ghost" size="sm" onClick={onCancelResourcePicker}>
              Close
            </Button>
            <Button
              type="button"
              size="sm"
              disabled={selectedCount === 0}
              onClick={onConfirmResources}
            >
              {preset?.id === "find_resource"
                ? `Add ${selectedCount || ""} resource${selectedCount === 1 ? "" : "s"}`
                : `Use ${selectedCount || ""} resource${selectedCount === 1 ? "" : "s"}`}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
