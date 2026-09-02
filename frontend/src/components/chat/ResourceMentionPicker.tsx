import { type CSSProperties, type MouseEvent, type ReactNode } from "react";

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

const KIND_LABEL_MAX = 5;

/** A short, mono kind chip: the extension when there is one, else the kind. */
export const resourceMentionKindLabel = (resource: ResourceRecord): string => {
  const name = String(resource.original_name ?? "");
  const extension = name.includes(".") ? name.slice(name.lastIndexOf(".") + 1) : "";
  const cleaned = extension.replace(/[^a-z0-9]/gi, "");
  if (cleaned.length > 0 && cleaned.length <= KIND_LABEL_MAX) {
    return cleaned.toUpperCase();
  }
  const kind = String(resource.resource_kind ?? "file").replace(/[^a-z0-9]/gi, "");
  return (kind || "file").slice(0, KIND_LABEL_MAX).toUpperCase();
};

export const resourceMentionOptionId = (listboxId: string, fileId: string): string =>
  `${listboxId}-${fileId.replace(/[^a-zA-Z0-9_-]/g, "_")}`;

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
      <strong key={index} className="brief-mention-match">
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
      className={cn("brief-mention-picker", `brief-mention-picker-${variant}`)}
      style={style}
      data-testid="brief-mention-picker"
      onMouseDown={preventBlur}
    >
      <div
        id={listboxId}
        role="listbox"
        aria-label="Files in your library"
        className="brief-mention-list"
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
              className={cn("brief-mention-option", active && "brief-mention-option-active")}
              onMouseEnter={() => onActivate(resource.file_id)}
              onClick={() => onPick(resource)}
            >
              <span className="brief-mention-kind">{resourceMentionKindLabel(resource)}</span>
              <span className="brief-mention-body">
                <span className="brief-mention-name">{emphasize(name, trimmedQuery)}</span>
                {meta ? <span className="brief-mention-meta">{meta}</span> : null}
              </span>
            </div>
          );
        })}
        {results.length === 0 ? (
          <div className="brief-mention-empty" role="presentation">
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
      <div className="brief-mention-footer">
        <span className="brief-mention-hint" aria-hidden="true">
          ↵ bring in · ↑↓ move · esc
        </span>
        {onUploadInstead ? (
          <button
            type="button"
            className="brief-mention-upload"
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
