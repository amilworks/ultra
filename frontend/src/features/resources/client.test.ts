import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import { loadComposerResources, loadLibraryResources } from "./client";
import {
  hasInternalResourcePath,
  resourceDisplayName,
  resourceOriginLabel,
} from "./presentation";

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
      limit: 50,
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

  it("keeps internal workspace paths out of resource display labels", () => {
    expect(
      resourceDisplayName({
        file_id: "file_1234567890abcdef",
        original_name: "/workspace/outputs/plot.png",
      })
    ).toBe("plot.png");
    expect(hasInternalResourcePath("/workspace/outputs/plot.png")).toBe(true);
  });

  it("summarizes resource origin without exposing raw URIs", () => {
    expect(
      resourceOriginLabel({
        source_type: "bisque_import",
        resource_kind: "image",
        source_uri: "https://bisque.example/data_service/image/123",
        client_view_url: "https://bisque.example/client_service/view?resource=123",
      })
    ).toBe("Imported BisQue image");
  });
});
