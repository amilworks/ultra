import { describe, expect, it, vi } from "vitest";

import { getCachedResourcePreview, getResourcePreview } from "./resourcePreviewCache";
import type { ResourceTextHead } from "../types";

function head(fileId: string): ResourceTextHead {
  return {
    file_id: fileId,
    original_name: `${fileId}.json`,
    content_type: "application/json",
    format: "json",
    total_size_bytes: 2,
    offset: 0,
    returned_bytes: 2,
    next_offset: 2,
    truncated: false,
    encoding: "utf-8",
    eol: "lf",
    line_count: 1,
    approx_total_lines: 1,
    text: "{}",
  };
}

describe("resourcePreviewCache", () => {
  it("dedupes concurrent + repeat requests for the same file to a single fetch", async () => {
    const fetcher = vi.fn(async (id: string) => head(id));
    const [a, b] = await Promise.all([
      getResourcePreview("dedupe-1", fetcher),
      getResourcePreview("dedupe-1", fetcher),
    ]);
    expect(a).toBe(b);
    expect(fetcher).toHaveBeenCalledTimes(1);

    // A later call returns the cached value without re-fetching.
    const c = await getResourcePreview("dedupe-1", fetcher);
    expect(c).toBe(a);
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(getCachedResourcePreview("dedupe-1")).toBe(a);
  });

  it("caps concurrency at 5", async () => {
    let active = 0;
    let peak = 0;
    const release: Array<() => void> = [];
    const fetcher = (id: string) =>
      new Promise<ResourceTextHead>((resolve) => {
        active += 1;
        peak = Math.max(peak, active);
        release.push(() => {
          active -= 1;
          resolve(head(id));
        });
      });

    const all = Promise.all(
      Array.from({ length: 12 }, (_, i) => getResourcePreview(`conc-${i}`, fetcher))
    );
    // Let the queue settle, then drain.
    await Promise.resolve();
    await new Promise((r) => setTimeout(r, 0));
    expect(peak).toBeLessThanOrEqual(5);
    while (release.length > 0) {
      release.shift()!();
      await new Promise((r) => setTimeout(r, 0));
    }
    await all;
    expect(peak).toBeLessThanOrEqual(5);
  });
});
