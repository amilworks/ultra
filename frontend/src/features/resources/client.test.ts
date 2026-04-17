import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import { loadComposerResources, loadLibraryResources } from "./client";

describe("resource slice client", () => {
  it("normalizes all-filters into backend-compatible query params", async () => {
    const apiClient = {
      listResources: vi.fn().mockResolvedValue({ count: 0, offset: 0, resources: [] }),
    } as unknown as ApiClient;

    await loadLibraryResources(apiClient, {
      query: "mitochondria",
      kind: "all",
      source: "all",
    });

    expect(apiClient.listResources).toHaveBeenCalledWith({
      kind: undefined,
      limit: 500,
      offset: 0,
      query: "mitochondria",
      source: undefined,
    });
  });

  it("keeps composer resource lookups lightweight by default", async () => {
    const apiClient = {
      listResources: vi.fn().mockResolvedValue({ count: 0, offset: 0, resources: [] }),
    } as unknown as ApiClient;

    await loadComposerResources(apiClient, { query: "atlas" });

    expect(apiClient.listResources).toHaveBeenCalledWith({
      limit: 200,
      query: "atlas",
    });
  });
});
