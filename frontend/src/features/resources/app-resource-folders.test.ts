import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");

describe("Resources app folder search wiring", () => {
  it("loads resource folders with the same debounced search query as files", () => {
    expect(appSource).toMatch(
      /loadResourceFolders\(apiClient,\s*\{\s*limit:\s*200,\s*query:\s*debouncedResourceQuery,\s*status:\s*resourceStatusFilter,\s*\}/s
    );
    expect(appSource).toMatch(
      /resourceCollectionRefreshToken,\s*resourceStatusFilter,\s*debouncedResourceQuery/s
    );
  });

  it("keeps the active folder context even when search filters the folder list", () => {
    expect(appSource).toContain("activeResourceCollectionSnapshot");
    expect(appSource).toMatch(
      /resourceCollections\.find\(\s*\(\s*collection\s*\)\s*=>\s*collection\.collection_id\s*===\s*activeResourceCollectionId\s*\)\s*\?\?\s*activeResourceCollectionSnapshot/s
    );
    expect(appSource).toMatch(/setActiveResourceCollectionSnapshot\(collection\)/);
  });

  it("adds newly uploaded Resources files to the active folder", () => {
    expect(appSource).toMatch(
      /const uploadTargetCollection\s*=\s*context\?\.uploadTargetCollection\s*\?\?\s*activeResourceCollection/s
    );
    expect(appSource).toMatch(
      /const activeUploadCollectionId\s*=\s*String\(uploadTargetCollection\?\.collection_id \?\? ""\)\.trim\(\)/s
    );
    expect(appSource).toMatch(/const response\s*=\s*await apiClient\.uploadFiles\(selectedFiles,/s);
    expect(appSource).toMatch(
      /await apiClient\.addResourcesToCollection\(\s*activeUploadCollectionId,\s*uploadedFileIds,\s*\{\s*source:\s*"resources_folder_upload",\s*\}\s*\)/s
    );
  });

  it("uses recoverable trash language for resource lifecycle confirmations", () => {
    expect(appSource).toContain("Move resource to trash?");
    expect(appSource).toContain("Move selected resources to trash?");
    expect(appSource).toContain("You can restore it from Deleted when needed.");
    expect(appSource).toContain("You can restore them from Deleted when needed.");
    expect(appSource).not.toContain("Delete uploaded resource?");
    expect(appSource).not.toContain("Delete selected resources?");
    expect(appSource).not.toContain("from your resource browser, BisQue catalog, and local cache?");
  });
});
