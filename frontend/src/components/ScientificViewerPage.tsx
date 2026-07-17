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
  return (
    <section className="mx-auto w-full flex-1 overflow-y-auto px-4 py-6 sm:px-6">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-4">
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
