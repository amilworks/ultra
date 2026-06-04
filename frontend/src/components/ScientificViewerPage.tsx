import { FolderOpen, Layers3 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
  onUseHdf5DatasetInChat?: (fileId: string, datasetPaths: string[]) => void;
  onOpenResources: () => void;
};

const formatFileCount = (count: number): string =>
  count === 1 ? "1 file" : `${count.toLocaleString()} files`;

export function ScientificViewerPage({
  uploadedFiles,
  bisqueLinksByFileId,
  apiClient,
  onUseHdf5DatasetInChat,
  onOpenResources,
}: ScientificViewerPageProps) {
  return (
    <section className="mx-auto w-full flex-1 overflow-y-auto px-4 py-6 sm:px-6">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div className="flex min-w-0 flex-col gap-1">
            <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
              <Layers3 className="size-4" aria-hidden="true" />
              <span>Scientific Viewer</span>
            </div>
            <h1 className="truncate text-2xl font-semibold tracking-normal text-foreground">
              Scientific Viewer
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline">{formatFileCount(uploadedFiles.length)}</Badge>
            <Button type="button" variant="outline" size="sm" onClick={onOpenResources}>
              <FolderOpen data-icon="inline-start" aria-hidden="true" />
              Resources
            </Button>
          </div>
        </div>

        <UploadViewerWorkspace
          uploadedFiles={uploadedFiles}
          bisqueLinksByFileId={bisqueLinksByFileId}
          apiClient={apiClient}
          onUseHdf5DatasetInChat={onUseHdf5DatasetInChat}
          className="viewer-workspace-embedded"
        />
      </div>
    </section>
  );
}
