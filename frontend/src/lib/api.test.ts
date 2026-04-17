import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClient } from "./api";

describe("ApiClient browser auth hardening", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("does not append api_key to browser-facing URLs", () => {
    const client = new ApiClient({
      baseUrl: "https://ultra.example.org",
      apiKey: "dev-secret",
    });

    const urls = [
      client.getBisqueOidcStartUrl("https://ultra.example.org/chat"),
      client.getBisqueBrowserLogoutUrl("https://ultra.example.org/"),
      client.resourceThumbnailUrl("file-123"),
      client.uploadPreviewUrl("file-123"),
      client.uploadDisplayUrl("file-123"),
      client.uploadSliceUrl("file-123", { axis: "z", z: 2 }),
      client.uploadAtlasUrl("file-123", { enhancement: "d", t: 1 }),
      client.uploadTileUrl("file-123", { axis: "z", level: 0, tileX: 0, tileY: 0 }),
      client.hdf5SlicePreviewUrl("file-123", { datasetPath: "/volume" }),
      client.hdf5AtlasPreviewUrl("file-123", { datasetPath: "/volume" }),
      client.artifactDownloadUrl("run-123", "reports/output.json"),
    ];

    urls.forEach((value) => {
      const parsed = new URL(value);
      expect(parsed.searchParams.has("api_key")).toBe(false);
    });
  });

  it("keeps header-based automation auth for fetch requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ authenticated: false }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
        },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({
      baseUrl: "https://ultra.example.org",
      apiKey: "dev-secret",
    });
    await client.getBisqueSession();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(init.headers).toMatchObject({
      "X-API-Key": "dev-secret",
    });
  });
});
