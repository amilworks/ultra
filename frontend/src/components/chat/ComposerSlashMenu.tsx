import { useEffect, useMemo, useRef } from "react";
import { Check, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { resourceMentionKindLabel } from "@/features/chat/resource-mention";
import { resourceDisplayName } from "@/features/resources/presentation";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ResourceRecord } from "@/types";

import type {
  ComposerWorkflowDefinition,
  ComposerWorkflowId,
  ComposerWorkflowPresetState,
} from "./composer-workflows";

/* The / menu and the library picker, inside the composer's card, in the one
   popover language the @ picker also speaks (see the composer's menu rules in
   styles.css). Deliberately dumb: the app owns the query, the active row and
   every keystroke; this component draws them and reports pointer intent. */

export type ComposerWorkflowGroup = {
  category: ComposerWorkflowDefinition["category"];
  items: ComposerWorkflowDefinition[];
};

type ComposerSlashMenuProps = {
  mode: "workflow" | "resource_picker";
  workflowGroups?: ComposerWorkflowGroup[];
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

/** The typeable shortcut shown in each row's margin — "/image", "/pro". The
    filter matches keywords, so a keyword is a truthful slug; the first one
    that no sibling workflow also claims is the one that names this row
    unambiguously ("/download", not a third "/bisque"). */
export const workflowSlugs = (
  workflows: readonly ComposerWorkflowDefinition[]
): Map<ComposerWorkflowId, string> => {
  const slugs = new Map<ComposerWorkflowId, string>();
  const claimed = new Map<string, number>();
  for (const workflow of workflows) {
    for (const keyword of workflow.keywords ?? []) {
      claimed.set(keyword, (claimed.get(keyword) ?? 0) + 1);
    }
  }
  for (const workflow of workflows) {
    const keywords = workflow.keywords ?? [];
    const own = keywords.find((keyword) => claimed.get(keyword) === 1) ?? keywords[0] ?? workflow.id;
    slugs.set(workflow.id, `/${own.replace(/\s+/g, "-").toLowerCase()}`);
  }
  return slugs;
};

const countLabel = (verb: string, count: number): string =>
  count > 0 ? `${verb} ${count} resource${count === 1 ? "" : "s"}` : `${verb} resources`;

const formatResourceDate = (value: string): string => {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) {
    return "";
  }
  return new Date(timestamp).toLocaleDateString([], { month: "short", day: "numeric" });
};

const ACTIVE_ROW_SELECTOR = '[data-composer-active="true"]';
const ROW_SCROLL_PADDING = 12;

const scrollActiveRowIntoView = (container: HTMLDivElement | null): void => {
  if (!container) {
    return;
  }
  const activeItem = container.querySelector<HTMLElement>(ACTIVE_ROW_SELECTOR);
  if (!activeItem) {
    return;
  }
  const containerRect = container.getBoundingClientRect();
  const activeItemRect = activeItem.getBoundingClientRect();
  const containerTop = container.scrollTop;
  const containerBottom = containerTop + container.clientHeight;
  const itemTop = activeItemRect.top - containerRect.top + container.scrollTop;
  const itemBottom = itemTop + activeItemRect.height;
  if (itemTop < containerTop + ROW_SCROLL_PADDING) {
    container.scrollTop = Math.max(itemTop - ROW_SCROLL_PADDING, 0);
    return;
  }
  if (itemBottom > containerBottom - ROW_SCROLL_PADDING) {
    container.scrollTop = itemBottom - container.clientHeight + ROW_SCROLL_PADDING;
  }
};

export function ComposerSlashMenu({
  mode,
  workflowGroups = [],
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

  const orderedWorkflows = useMemo(
    () => workflowGroups.flatMap((group) => group.items),
    [workflowGroups]
  );
  const slugs = useMemo(() => workflowSlugs(orderedWorkflows), [orderedWorkflows]);

  useEffect(() => {
    if (mode !== "workflow" || !activeWorkflowId) {
      return;
    }
    scrollActiveRowIntoView(workflowListRef.current);
  }, [mode, activeWorkflowId, orderedWorkflows]);

  useEffect(() => {
    if (mode !== "resource_picker" || !activeResourceId) {
      return;
    }
    scrollActiveRowIntoView(resourceListRef.current);
  }, [mode, activeResourceId, resources]);

  if (mode === "workflow") {
    const count = orderedWorkflows.length;
    return (
      <div className="composer-menu composer-slash-menu" data-testid="composer-slash-menu">
        <div
          ref={workflowListRef}
          className="composer-menu-list"
          role="listbox"
          aria-label="Workflows"
        >
          {count === 0 ? (
            <div className="composer-menu-empty" role="presentation">
              No workflow matches that.
            </div>
          ) : null}
          {workflowGroups.map((group) => (
            <div
              key={group.category}
              className="composer-menu-group"
              role="group"
              aria-label={group.category}
            >
              <div className="composer-menu-eyebrow" aria-hidden="true">
                {group.category}
              </div>
              {group.items.map((workflow) => {
                const Icon = workflow.icon;
                const comingSoon = workflow.comingSoon === true;
                const active = !comingSoon && workflow.id === activeWorkflowId;
                const selectWorkflow = () => {
                  if (!comingSoon) {
                    onSelectWorkflow?.(workflow);
                  }
                };
                return (
                  <div
                    key={workflow.id}
                    role="option"
                    aria-selected={active}
                    aria-disabled={comingSoon || undefined}
                    data-testid={`composer-workflow-${workflow.id}`}
                    data-composer-active={active ? "true" : undefined}
                    className={cn(
                      "composer-menu-row",
                      active && "composer-menu-row-active",
                      comingSoon && "composer-menu-row-disabled"
                    )}
                    onMouseDown={(event) => {
                      // Select on mousedown, before the click can blur the editor.
                      event.preventDefault();
                      selectWorkflow();
                    }}
                    onClick={selectWorkflow}
                  >
                    <Icon className="composer-menu-icon" aria-hidden="true" />
                    <div className="composer-menu-body">
                      <span className="composer-menu-title">{workflow.label}</span>
                      <span className="composer-menu-detail">{workflow.description}</span>
                    </div>
                    <span className="composer-menu-aside">
                      {comingSoon ? "soon" : slugs.get(workflow.id)}
                    </span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        <div className="composer-menu-footer">
          <span className="composer-menu-hint" aria-hidden="true">
            ↵ choose · ↑↓ move · esc
          </span>
          <span>{`${count} workflow${count === 1 ? "" : "s"}`}</span>
        </div>
      </div>
    );
  }

  const selectedCount = selectedResourceIds.size;
  const finding = preset?.id === "find_resource";
  const resourceTitle = finding
    ? "Find resources"
    : `Choose resources for ${preset?.label ?? "this workflow"}`;
  const resourceDescription = finding
    ? "Search your library and stage files into this chat."
    : "Stage one or more files before sending the workflow-backed prompt.";

  return (
    <div
      className="composer-menu composer-resource-picker"
      data-testid="composer-resource-picker"
    >
      <div className="composer-menu-header">
        <div className="composer-menu-body">
          <span className="composer-menu-title">{resourceTitle}</span>
          <span className="composer-menu-detail">{resourceDescription}</span>
        </div>
        {selectedCount > 0 ? (
          <span className="composer-menu-aside">{`${selectedCount} selected`}</span>
        ) : null}
      </div>
      <input
        autoFocus
        type="text"
        className="composer-menu-search"
        value={resourceQuery}
        onChange={(event) => onResourceQueryChange?.(event.target.value)}
        onKeyDown={onResourceInputKeyDown}
        placeholder="Search files, BisQue IDs, or URLs"
        aria-label="Find resources"
        autoComplete="off"
        spellCheck={false}
      />
      <div
        ref={resourceListRef}
        className="composer-menu-list"
        role="listbox"
        aria-label="Resources"
        aria-multiselectable="true"
      >
        {resourcesError ? (
          <div className="composer-menu-empty composer-menu-error" role="presentation">
            {resourcesError}
          </div>
        ) : null}
        {resourcesLoading ? (
          <div className="composer-menu-empty" role="presentation">
            <Loader2 className="composer-menu-icon composer-menu-spin" aria-hidden="true" />
            Searching your library…
          </div>
        ) : null}
        {!resourcesLoading && !resourcesError && resources.length === 0 ? (
          <div className="composer-menu-empty" role="presentation">
            {resourceQuery.trim()
              ? `Nothing in your library matches “${resourceQuery.trim()}”.`
              : "Your library is empty."}
          </div>
        ) : null}
        {!resourcesLoading && !resourcesError
          ? resources.map((resource) => {
              const selected = selectedResourceIds.has(resource.file_id);
              const active = resource.file_id === activeResourceId;
              const meta = [formatBytes(resource.size_bytes), formatResourceDate(resource.created_at)]
                .filter((part) => part && part.length > 0)
                .join(" · ");
              return (
                <div
                  key={resource.file_id}
                  role="option"
                  aria-selected={selected}
                  data-composer-active={active ? "true" : undefined}
                  className={cn(
                    "composer-menu-row",
                    active && "composer-menu-row-active",
                    selected && "composer-menu-row-selected"
                  )}
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => onToggleResource?.(resource)}
                >
                  <span
                    className={cn("composer-menu-check", selected && "composer-menu-check-on")}
                    aria-hidden="true"
                  >
                    <Check />
                  </span>
                  <span className="composer-menu-kind">{resourceMentionKindLabel(resource)}</span>
                  <div className="composer-menu-body">
                    <span className="composer-menu-title">{resourceDisplayName(resource)}</span>
                    {resource.source_uri ? (
                      <span className="composer-menu-detail">{resource.source_uri}</span>
                    ) : null}
                  </div>
                  {meta ? <span className="composer-menu-aside">{meta}</span> : null}
                </div>
              );
            })
          : null}
      </div>
      <div className="composer-menu-footer">
        <span className="composer-menu-hint" aria-hidden="true">
          ↵ select · ↑↓ move · esc
        </span>
        <span className="composer-menu-actions">
          <Button type="button" variant="ghost" size="sm" onClick={onCancelResourcePicker}>
            Close
          </Button>
          <Button
            type="button"
            size="sm"
            disabled={selectedCount === 0}
            onClick={onConfirmResources}
          >
            {countLabel(finding ? "Add" : "Use", selectedCount)}
          </Button>
        </span>
      </div>
    </div>
  );
}
