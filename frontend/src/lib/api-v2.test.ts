import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiV2Client } from "./api-v2";

describe("ApiV2Client threads", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("preserves thread pagination metadata from the V2 contract", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/threads?limit=2&offset=4&status=active") {
        return new Response(
          JSON.stringify({
            count: 2,
            total_count: 9,
            limit: 2,
            offset: 4,
            has_more: true,
            threads: [
              {
                thread_id: "thread_v2_1",
                title: "first saved run",
                status: "active",
                created_at: "2026-06-03T10:00:00Z",
                updated_at: "2026-06-03T10:01:00Z",
                metadata: {},
              },
              {
                thread_id: "thread_v2_2",
                title: "second saved run",
                status: "active",
                created_at: "2026-06-03T09:00:00Z",
                updated_at: "2026-06-03T09:01:00Z",
                metadata: {},
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiV2Client({ baseUrl: "https://ultra.example.org" });
    const response = await client.listThreads({ limit: 2, offset: 4, status: "active" });

    expect(response.count).toBe(2);
    expect(response.total_count).toBe(9);
    expect(response.limit).toBe(2);
    expect(response.offset).toBe(4);
    expect(response.has_more).toBe(true);
    expect(response.threads.map((thread) => thread.thread_id)).toEqual(["thread_v2_1", "thread_v2_2"]);
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads?limit=2&offset=4&status=active",
    ]);
  });
});
