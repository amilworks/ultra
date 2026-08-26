import { Layers3 } from "lucide-react";

import type { ApiClient } from "@/lib/api";
import { BrandWordmark } from "./BrandWordmark";
import { Button } from "./ui/button";
import type { UploadedFileRecord } from "../types";
import {
  UploadViewerWorkspace,
  type BisqueViewerLink,
} from "./UploadViewerSheet";

type ScientificViewerPageProps = {
  uploadedFiles: UploadedFileRecord[];
  bisqueLinksByFileId: Record<string, BisqueViewerLink>;
  apiClient: ApiClient;
  // File ids a deep link or chat link asked for that the catalog answered
  // 404/403/410 for. The backend answers 404 for both "removed" and "not
  // yours", so the copy deliberately names both without guessing which.
  unavailableFileIds?: string[];
  // File ids whose lookup failed for a transient reason (5xx, network).
  // Retryable, so every surface that mentions them offers Retry.
  failedFileIds?: string[];
  onOpenResources?: () => void;
  // Switch to the chat panel; offered in place of "Go back" when this tab has
  // no history entry to return to (a deep link opened in a fresh tab).
  onOpenChat?: () => void;
  // Ask for the same file ids again.
  onRetry?: () => void;
};

type LensNoticeVariant = "unavailable" | "failed";

const LENS_NOTICE_COPY: Record<LensNoticeVariant, { title: string; body: string }> = {
  unavailable: {
    title: "This resource isn't available",
    body: "It may have been removed, or it isn't shared with you.",
  },
  failed: {
    title: "This resource couldn't be loaded",
    body: "Check your connection and try again.",
  },
};

// "Go back" only when the browser has somewhere to go back to. history.length
// is 1 in a fresh tab, where Back would do nothing and the button would be a lie.
const canGoBack = (): boolean => typeof window !== "undefined" && window.history.length > 1;

function LensEmptyNotice({
  variant,
  onOpenResources,
  onOpenChat,
  onRetry,
}: {
  variant: LensNoticeVariant;
  onOpenResources?: () => void;
  onOpenChat?: () => void;
  onRetry?: () => void;
}) {
  const copy = LENS_NOTICE_COPY[variant];
  // Calm by design: no alert role, no red. Arriving here is an ordinary outcome
  // of following a stale link or a flaky connection, not an error the user caused.
  return (
    <section
      aria-labelledby="lens-notice-heading"
      className="mx-auto flex w-full max-w-md flex-col items-center gap-2 rounded-xl px-6 py-10 text-center"
      style={{ background: "var(--bg-sunk)" }}
    >
      <h2 id="lens-notice-heading" className="m-0 text-base font-semibold text-foreground">
        {copy.title}
      </h2>
      <p className="m-0 text-sm" style={{ color: "var(--text-muted)" }}>
        {copy.body}
      </p>
      <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
        {variant === "failed" ? (
          <Button type="button" variant="ghost" size="sm" onClick={() => onRetry?.()}>
            Retry
          </Button>
        ) : null}
        <Button type="button" variant="ghost" size="sm" onClick={() => onOpenResources?.()}>
          Open Resources
        </Button>
        {canGoBack() ? (
          <Button type="button" variant="ghost" size="sm" onClick={() => window.history.back()}>
            Go back
          </Button>
        ) : (
          <Button type="button" variant="ghost" size="sm" onClick={() => onOpenChat?.()}>
            Open chat
          </Button>
        )}
      </div>
    </section>
  );
}

const filesOutOf = (count: number, total: number): string => `${count} of ${total} files`;

// One quiet line for the case where SOME of the requested files opened: the
// workspace stays, the shortfall is named, and a transient failure offers Retry.
function LensPartialStatus({
  openCount,
  unavailableCount,
  failedCount,
  onRetry,
}: {
  openCount: number;
  unavailableCount: number;
  failedCount: number;
  onRetry?: () => void;
}) {
  if (openCount === 0 || unavailableCount + failedCount === 0) {
    return null;
  }
  const total = openCount + unavailableCount + failedCount;
  return (
    <p
      role="status"
      className="m-0 flex min-w-0 flex-wrap items-center gap-x-2 text-xs"
      style={{ color: "var(--text-muted)" }}
    >
      {unavailableCount > 0 ? (
        <span>
          {`${filesOutOf(unavailableCount, total)} ${unavailableCount === 1 ? "isn't" : "aren't"} available`}
        </span>
      ) : null}
      {unavailableCount > 0 && failedCount > 0 ? <span aria-hidden="true">·</span> : null}
      {failedCount > 0 ? (
        <>
          <span>{`${filesOutOf(failedCount, total)} couldn't be loaded`}</span>
          <span aria-hidden="true">·</span>
          <Button
            type="button"
            variant="link"
            size="xs"
            className="h-auto p-0 text-xs"
            onClick={() => onRetry?.()}
          >
            Retry
          </Button>
        </>
      ) : null}
    </p>
  );
}

export function ScientificViewerPage({
  uploadedFiles,
  bisqueLinksByFileId,
  apiClient,
  unavailableFileIds = [],
  failedFileIds = [],
  onOpenResources,
  onOpenChat,
  onRetry,
}: ScientificViewerPageProps) {
  // Nothing opened but something was asked for: say why, in place of the workspace.
  // A transient failure outranks "unavailable" because Retry might still fix it.
  const emptyNoticeVariant: LensNoticeVariant | null =
    uploadedFiles.length > 0
      ? null
      : failedFileIds.length > 0
        ? "failed"
        : unavailableFileIds.length > 0
          ? "unavailable"
          : null;
  // flex-col section so the inner column can stretch to fill the viewport height
  // (viewers size themselves via flex:1); overflow-y-auto still scrolls a short
  // viewport that can't fit the content.
  return (
    <section className="mx-auto flex w-full flex-1 flex-col overflow-y-auto px-4 py-6 sm:px-6">
      {/* Wide cap (not max-w-7xl): the carpet plot is a very wide-format matrix
          (hundreds–thousands of time frames), so it reads far better using the
          available width on large screens instead of sitting in a 1280px column
          with empty margins. Still bounded so it never stretches absurdly wide.
          flex-1 + min-h-0 lets the viewer fill the height instead of collapsing
          to its min-height and leaving empty space below. */}
      <div className="mx-auto flex w-full max-w-[1800px] flex-1 flex-col gap-4 min-h-0">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          {/* A compact breadcrumb preserves the calm viewer chrome while the current
              page remains a real h1 for document hierarchy and assistive technology. */}
          <nav
            aria-label="Breadcrumb"
            className="flex min-w-0 items-center gap-2 text-sm text-muted-foreground"
          >
            <Layers3 className="size-4 shrink-0" aria-hidden="true" />
            <ol className="flex min-w-0 items-center gap-2">
              <li className="min-w-0">
                <BrandWordmark className="truncate" />
              </li>
              <li aria-hidden="true" className="text-muted-foreground/50">
                /
              </li>
              <li aria-current="page" className="min-w-0">
                <h1 className="m-0 truncate text-sm font-semibold text-foreground">
                  Lens
                </h1>
              </li>
            </ol>
          </nav>
          <LensPartialStatus
            openCount={uploadedFiles.length}
            unavailableCount={unavailableFileIds.length}
            failedCount={failedFileIds.length}
            onRetry={onRetry}
          />
        </div>

        {emptyNoticeVariant ? (
          <LensEmptyNotice
            variant={emptyNoticeVariant}
            onOpenResources={onOpenResources}
            onOpenChat={onOpenChat}
            onRetry={onRetry}
          />
        ) : (
          <UploadViewerWorkspace
            uploadedFiles={uploadedFiles}
            bisqueLinksByFileId={bisqueLinksByFileId}
            apiClient={apiClient}
            className="viewer-workspace-embedded"
          />
        )}
      </div>
    </section>
  );
}
