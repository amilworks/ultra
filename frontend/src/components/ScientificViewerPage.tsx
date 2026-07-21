import { Layers3 } from "lucide-react";

import type { ApiClient } from "@/lib/api";
import type { UploadedFileRecord } from "../types";
import {
  UploadViewerWorkspace,
  type BisqueViewerLink,
} from "./UploadViewerSheet";

type ScientificViewerPageProps = {
  uploadedFiles: UploadedFileRecord[];
  bisqueLinksByFileId: Record<string, BisqueViewerLink>;
  apiClient: ApiClient;
};

export function ScientificViewerPage({
  uploadedFiles,
  bisqueLinksByFileId,
  apiClient,
}: ScientificViewerPageProps) {
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
          {/* One calm breadcrumb-as-title row (no oversized H1): product name recedes
              to muted, the viewer name leads via weight + color-tier — the system's
              hierarchy convention, no accent hue. Reclaims the vertical space the H1
              took above the canvas. */}
          <div className="flex min-w-0 items-center gap-2 text-sm text-muted-foreground">
            <Layers3 className="size-4 shrink-0" aria-hidden="true" />
            <span className="truncate">BisQue Ultra</span>
            <span aria-hidden="true" className="text-muted-foreground/50">/</span>
            <span className="truncate font-semibold text-foreground">Lens</span>
          </div>
        </div>

        <UploadViewerWorkspace
          uploadedFiles={uploadedFiles}
          bisqueLinksByFileId={bisqueLinksByFileId}
          apiClient={apiClient}
          className="viewer-workspace-embedded"
        />
      </div>
    </section>
  );
}
