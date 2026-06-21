import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiClient, ApiError } from "./api";

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const THREAD = {
  thread_id: "thread_hyd",
  title: "Plot the data",
  status: "active",
  latest_run_id: "run_hyd",
  created_at: "2026-06-20T00:00:00Z",
  updated_at: "2026-06-20T00:00:05Z",
  // No client snapshot in metadata -> hydration rebuilds from durable messages.
  metadata: { conversation_id: "thread_hyd" },
};

const USER_MESSAGE = {
  message_id: "m1",
  role: "user",
  content: "Plot the data",
  created_at: "2026-06-20T00:00:00Z",
  run_id: "run_hyd",
};

const ASSISTANT_MESSAGE = {
  message_id: "m2",
  role: "assistant",
  content: "Here is the plot.",
  created_at: "2026-06-20T00:00:05Z",
  run_id: "run_hyd",
};

const RUN = {
  run_id: "run_hyd",
  status: "succeeded",
  response_text: "Here is the plot.",
  updated_at: "2026-06-20T00:00:05Z",
  metadata: {},
};

const stateMessagesOf = (record: { state?: unknown }): Array<Record<string, unknown>> => {
  const state = record.state as { messages?: unknown } | undefined;
  return Array.isArray(state?.messages) ? (state?.messages as Array<Record<string, unknown>>) : [];
};

describe("conversation hydration durability", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("restores both the user and assistant turns from durable thread messages", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v2/threads/thread_hyd")) return jsonResponse(THREAD);
      if (url.endsWith("/v2/threads/thread_hyd/messages")) {
        return jsonResponse({ messages: [USER_MESSAGE, ASSISTANT_MESSAGE] });
      }
      if (url.endsWith("/v2/runs/run_hyd")) return jsonResponse(RUN);
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const record = await client.getConversation("thread_hyd");

    const roles = stateMessagesOf(record).map((message) => message.role);
    expect(roles).toContain("user");
    expect(roles).toContain("assistant");
    expect(stateMessagesOf(record).find((m) => m.role === "user")?.content).toBe("Plot the data");
  });

  it("refuses to drop the user turn when the durable messages fetch fails and there is no snapshot", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v2/threads/thread_hyd")) return jsonResponse(THREAD);
      if (url.endsWith("/v2/threads/thread_hyd/messages")) {
        return jsonResponse({ detail: "messages unavailable" }, 500);
      }
      if (url.endsWith("/v2/runs/run_hyd")) return jsonResponse(RUN);
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    // Previously this returned an assistant-only transcript (user message lost).
    // Now it surfaces a recoverable error instead of overwriting state with a
    // lossy rebuild.
    await expect(client.getConversation("thread_hyd")).rejects.toBeInstanceOf(ApiError);

    const messageFetches = fetchMock.mock.calls.filter(([input]) =>
      String(input).endsWith("/messages")
    );
    expect(messageFetches.length).toBe(3); // initial + 2 retries
  });

  it("retries a transient messages-fetch failure and recovers the full transcript", async () => {
    let messageAttempts = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v2/threads/thread_hyd")) return jsonResponse(THREAD);
      if (url.endsWith("/v2/threads/thread_hyd/messages")) {
        messageAttempts += 1;
        if (messageAttempts < 2) {
          return jsonResponse({ detail: "temporary" }, 503);
        }
        return jsonResponse({ messages: [USER_MESSAGE, ASSISTANT_MESSAGE] });
      }
      if (url.endsWith("/v2/runs/run_hyd")) return jsonResponse(RUN);
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const record = await client.getConversation("thread_hyd");

    expect(messageAttempts).toBe(2);
    const roles = stateMessagesOf(record).map((message) => message.role);
    expect(roles).toContain("user");
    expect(roles).toContain("assistant");
  });
});
