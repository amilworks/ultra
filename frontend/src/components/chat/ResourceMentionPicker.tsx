import { type CSSProperties, type MouseEvent, type ReactNode } from "react";

import { resourceMentionKindLabel, resourceMentionOptionId } from "@/features/chat/resource-mention";
import { findQueryMatches, resourceDisplayName } from "@/features/resources/presentation";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ResourceRecord } from "@/types";

/**
 * The @ picker: the user's library, opened at the caret.
 *
 * Deliberately a dumb listbox. The composer owns the query (it IS the text
 * after the @), the results, the active row, and every keystroke; this
 * component only draws them and reports pointer intent. On desktop it floats
 * at the caret; on phones it lays under the line as a sheet.
 */

export type ResourceMentionPickerProps = {
  variant: "popover" | "sheet";
  /** Popover only: horizontal offset within the positioned parent, in px. The
   *  popover always opens UPWARD from the composer (CSS anchors its bottom), so
   *  no vertical offset is needed and nothing can clip under the actions row. */
  anchor?: { left: number } | null;
  listboxId: string;
  query: string;
  results: readonly ResourceRecord[];
  loading: boolean;
  error?: string | null;
  activeFileId: string | null;
  onActivate: (fileId: string) => void;
  onPick: (resource: ResourceRecord) => void;
  onUploadInstead?: () => void;
};

const formatMentionDate = (value: string): string => {
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "";
    }
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
  } catch {
    return "";
  }
};

const emphasize = (text: string, query: string): ReactNode => {
  const ranges = findQueryMatches(text, query);
  if (ranges.length === 0) {
    return text;
  }
  const parts: ReactNode[] = [];
  let cursor = 0;
  ranges.forEach((range, index) => {
    if (range.start > cursor) {
      parts.push(text.slice(cursor, range.start));
    }
    parts.push(
      <strong key={index} className="composer-mention-match">
        {text.slice(range.start, range.end)}
      </strong>
    );
    cursor = range.end;
  });
  if (cursor < text.length) {
    parts.push(text.slice(cursor));
  }
  return parts;
};

export function ResourceMentionPicker({
  variant,
  anchor,
  listboxId,
  query,
  results,
  loading,
  error,
  activeFileId,
  onActivate,
  onPick,
  onUploadInstead,
}: ResourceMentionPickerProps) {
  const style: CSSProperties | undefined =
    variant === "popover" && anchor ? { left: anchor.left } : undefined;
  const preventBlur = (event: MouseEvent) => event.preventDefault();
  const trimmedQuery = query.trim();

  return (
    <div
      className={cn("composer-menu composer-mention-picker", `composer-mention-picker-${variant}`)}
      style={style}
      data-testid="composer-mention-picker"
      onMouseDown={preventBlur}
    >
      <div
        id={listboxId}
        role="listbox"
        aria-label="Files in your library"
        className="composer-menu-list"
      >
        {results.map((resource) => {
          const active = resource.file_id === activeFileId;
          const name = resourceDisplayName(resource);
          const meta = [formatBytes(resource.size_bytes), formatMentionDate(resource.created_at)]
            .filter((part) => part && part.length > 0)
            .join(" · ");
          return (
            <div
              key={resource.file_id}
              id={resourceMentionOptionId(listboxId, resource.file_id)}
              role="option"
              aria-selected={active}
              className={cn("composer-menu-row", active && "composer-menu-row-active")}
              onMouseEnter={() => onActivate(resource.file_id)}
              onClick={() => onPick(resource)}
            >
              <span className="composer-menu-kind">{resourceMentionKindLabel(resource)}</span>
              <span className="composer-menu-body">
                <span className="composer-menu-title">{emphasize(name, trimmedQuery)}</span>
              </span>
              {meta ? <span className="composer-menu-aside">{meta}</span> : null}
            </div>
          );
        })}
        {results.length === 0 ? (
          <div className="composer-menu-empty" role="presentation">
            {error
              ? "Your library could not be searched right now."
              : loading
                ? "Searching your library…"
                : trimmedQuery
                  ? `Nothing in your library matches “${trimmedQuery}”.`
                  : "Your library is empty."}
          </div>
        ) : null}
      </div>
      <div className="composer-menu-footer">
        <span className="composer-menu-hint" aria-hidden="true">
          ↵ bring in · ↑↓ move · esc
        </span>
        {onUploadInstead ? (
          <button
            type="button"
            className="composer-menu-foot-action"
            onClick={onUploadInstead}
            tabIndex={-1}
          >
            Upload instead…
          </button>
        ) : null}
      </div>
    </div>
  );
}
