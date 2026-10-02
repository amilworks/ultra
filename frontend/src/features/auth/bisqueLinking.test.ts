import { describe, expect, it, vi } from "vitest";

import { loadOptionalBisqueImageCount } from "./bisqueLinking";

describe("loadOptionalBisqueImageCount", () => {
  it("keeps an authoritative link success when the optional count lookup fails", async () => {
    const searchBisqueResources = vi.fn(async () => {
      throw new Error("image-count service unavailable");
    });

    await expect(loadOptionalBisqueImageCount(searchBisqueResources)).resolves.toBeNull();
    expect(searchBisqueResources).toHaveBeenCalledTimes(1);
    expect(searchBisqueResources).toHaveBeenCalledWith({
      resourceType: "image",
      scope: "owner",
      limit: 1,
      countAll: true,
    });
  });

  it("returns the linked account's image count when enrichment succeeds", async () => {
    const searchBisqueResources = vi.fn(async () => ({ count: 142, results: [] }));

    await expect(loadOptionalBisqueImageCount(searchBisqueResources)).resolves.toBe(142);
  });
});
