import { readFileSync } from "node:fs";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

  it("builds uploaded image slice URLs through the V2 upload API", () => {
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    expect(client.uploadSliceUrl("file-123", { axis: "z", z: 2 })).toBe(
      "https://ultra.example.org/v2/uploads/file-123/slice?axis=z&z=2"
    );
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

describe("ApiClient V2 chat bridge", () => {
  const createMemoryStorage = (): Storage => {
    const values = new Map<string, string>();
    return {
      get length() {
        return values.size;
      },
      clear: () => values.clear(),
      getItem: (key: string) => values.get(key) ?? null,
      key: (index: number) => Array.from(values.keys())[index] ?? null,
      removeItem: (key: string) => {
        values.delete(key);
      },
      setItem: (key: string, value: string) => {
        values.set(key, String(value));
      },
    };
  };

  const browserStorage = (): Storage => window.localStorage;

  beforeEach(() => {
    const storage = createMemoryStorage();
    Object.defineProperty(window, "localStorage", {
      value: storage,
      configurable: true,
    });
    vi.stubGlobal("localStorage", storage);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("streams chat through V2 runs instead of legacy V1 chat endpoints", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url.endsWith("/v2/threads")) {
        return new Response(
          JSON.stringify({
            thread_id: "thread_v2_123",
            title: "create a plot",
            status: "active",
            created_at: "2026-05-31T00:00:00Z",
            updated_at: "2026-05-31T00:00:00Z",
            metadata: {},
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith("/v2/threads/thread_v2_123/runs")) {
        return new Response(
          JSON.stringify({
            run_id: "run_v2_123",
            thread_id: "thread_v2_123",
            goal: "create a plot",
            status: "queued",
            workflow_kind: "deepagents",
            created_at: "2026-05-31T00:00:00Z",
            updated_at: "2026-05-31T00:00:00Z",
            metadata: {},
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.includes("/v2/runs/run_v2_123/events") && url.includes("stream=true")) {
        const body = [
          'event: run_event\ndata: {"run_id":"run_v2_123","thread_id":"thread_v2_123","event_kind":"message.delta","level":"info","payload":{"delta":"Hello"}}\n\n',
          'event: run_event\ndata: {"run_id":"run_v2_123","thread_id":"thread_v2_123","event_kind":"run.completed","level":"info","payload":{"response_text":"Hello"}}\n\n',
        ].join("");
        return new Response(encoder.encode(body), {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      if (url.endsWith("/v2/runs/run_v2_123")) {
        return new Response(
          JSON.stringify({
            run_id: "run_v2_123",
            thread_id: "thread_v2_123",
            goal: "create a plot",
            status: "succeeded",
            workflow_kind: "deepagents",
            response_text: "Hello",
            created_at: "2026-05-31T00:00:00Z",
            updated_at: "2026-05-31T00:00:01Z",
            metadata: {},
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const tokens: string[] = [];
    const runEvents: string[] = [];
    const runStarts: string[] = [];

    const response = await client.chatStream(
      {
        messages: [{ role: "user", content: "create a plot" }],
        uploaded_files: [],
        conversation_id: "conversation-local-123",
        goal: "create a plot",
        budgets: { max_tool_calls: 1, max_runtime_seconds: 60 },
      },
      {
        onToken: (delta) => tokens.push(delta),
        onRunStarted: ({ runId }) => runStarts.push(runId),
        onRunEvent: (event) => runEvents.push(event.event_type),
      }
    );

    expect(response.run_id).toBe("run_v2_123");
    expect(response.response_text).toBe("Hello");
    expect(tokens).toEqual(["Hello"]);
    expect(runStarts).toEqual(["run_v2_123"]);
    expect(runEvents).toEqual(["message.delta", "run.completed"]);
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads",
      "https://ultra.example.org/v2/threads/thread_v2_123/runs",
      "https://ultra.example.org/v2/runs/run_v2_123/events?stream=true&after_sequence=0",
      "https://ultra.example.org/v2/runs/run_v2_123",
    ]);
  });

  it("resumes an existing V2 run stream after the last hydrated event sequence", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (
        url ===
        "https://ultra.example.org/v2/runs/run_resume/events?stream=true&after_sequence=7"
      ) {
        const body = [
          'event: run_event\ndata: {"run_id":"run_resume","event_kind":"message.delta","sequence":8,"payload":{"text":" more"}}\n\n',
          'event: run_event\ndata: {"run_id":"run_resume","event_kind":"run.completed","sequence":9,"payload":{"response_text":"done more"}}\n\n',
        ].join("");
        return new Response(encoder.encode(body), {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      if (url === "https://ultra.example.org/v2/runs/run_resume") {
        return new Response(
          JSON.stringify({
            run_id: "run_resume",
            status: "succeeded",
            response_text: "done more",
            updated_at: "2026-05-31T00:00:01Z",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const tokens: string[] = [];
    const eventSequences: Array<unknown> = [];
    const response = await client.resumeRunStream("run_resume", {
      afterSequence: 7,
      onToken: (delta) => tokens.push(delta),
      onRunEvent: (event) => eventSequences.push(event.payload?.sequence),
    });

    expect(response.response_text).toBe("done more");
    expect(tokens).toEqual([" more"]);
    expect(eventSequences).toEqual([8, 9]);
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/runs/run_resume/events?stream=true&after_sequence=7",
      "https://ultra.example.org/v2/runs/run_resume",
    ]);
  });

  it("resumes an existing V2 run stream from the beginning with an explicit zero cursor", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (
        url ===
        "https://ultra.example.org/v2/runs/run_resume/events?stream=true&after_sequence=0"
      ) {
        const body = [
          'event: run_event\ndata: {"run_id":"run_resume","event_kind":"message.delta","sequence":1,"payload":{"text":"full"}}\n\n',
          'event: run_event\ndata: {"run_id":"run_resume","event_kind":"run.completed","sequence":2,"payload":{"response_text":"full answer"}}\n\n',
        ].join("");
        return new Response(encoder.encode(body), {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      if (url === "https://ultra.example.org/v2/runs/run_resume") {
        return new Response(
          JSON.stringify({
            run_id: "run_resume",
            status: "succeeded",
            response_text: "full answer",
            updated_at: "2026-05-31T00:00:01Z",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.resumeRunStream("run_resume");

    expect(response.response_text).toBe("full answer");
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/runs/run_resume/events?stream=true&after_sequence=0",
      "https://ultra.example.org/v2/runs/run_resume",
    ]);
  });

  it("generates chat titles locally instead of calling legacy V1 title endpoint", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.chatTitle({
      messages: [{ role: "user", content: "create a matplotlib y = x^2 plot" }],
      max_words: 4,
    });

    expect(response).toEqual({
      title: "create a matplotlib y",
      model: "frontend-local",
      strategy: "fallback",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("sends a stable idempotency key to V2 run creation", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url.endsWith("/v2/threads")) {
        return new Response(JSON.stringify({ thread_id: "thread_v2_123" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith("/v2/threads/thread_v2_123/runs")) {
        return new Response(JSON.stringify({ run_id: "run_v2_123" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/v2/runs/run_v2_123/events")) {
        return new Response(
          encoder.encode(
            'event: run_event\ndata: {"run_id":"run_v2_123","thread_id":"thread_v2_123","event_kind":"run.completed","payload":{"response_text":"done"}}\n\n'
          ),
          { status: 200, headers: { "Content-Type": "text/event-stream" } }
        );
      }
      if (url.endsWith("/v2/runs/run_v2_123")) {
        return new Response(JSON.stringify({ run_id: "run_v2_123", response_text: "done" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    await client.chatStream({
      messages: [{ role: "user", content: "retry-safe prompt" }],
      uploaded_files: [],
      conversation_id: "conversation-local-123",
      goal: "retry-safe prompt",
      idempotency_key: "message-key-123",
    });

    const runCreateCall = fetchMock.mock.calls.find(([input]) =>
      String(input).endsWith("/v2/threads/thread_v2_123/runs")
    );
    expect(runCreateCall).toBeTruthy();
    if (!runCreateCall) {
      throw new Error("expected V2 run creation call");
    }
    const [, init] = runCreateCall;
    if (!init) {
      throw new Error("expected V2 run creation request init");
    }
    const headers = new Headers(init.headers);
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("idempotency-key")).toBe("message-key-123");
    expect(JSON.parse(String(init.body))).toMatchObject({
      idempotency_key: "message-key-123",
    });
  });

  it("uploads local chat files through V2 without probing legacy upload routes", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/uploads") {
        expect(init?.method).toBe("POST");
        expect(init?.body).toBeInstanceOf(FormData);
        return new Response(
          JSON.stringify({
            file_count: 1,
            uploaded: [
              {
                file_id: "file_v2_image",
                original_name: "prairie.jpg",
                content_type: "image/jpeg",
                size_bytes: 4,
                sha256: "abc123",
                created_at: "2026-05-31T00:00:00Z",
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.uploadFiles([
      new File(["data"], "prairie.jpg", { type: "image/jpeg" }),
    ]);

    expect(response.uploaded[0].file_id).toBe("file_v2_image");
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/uploads",
    ]);
  });

  it("lists resources through V2 without probing legacy resource routes", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/resources?limit=50&offset=0&q=prairie&kind=image") {
        return new Response(
          JSON.stringify({
            count: 1,
            resources: [
              {
                file_id: "file_v2_image",
                original_name: "prairie.jpg",
                content_type: "image/jpeg",
                size_bytes: 4,
                sha256: "abc123",
                created_at: "2026-05-31T00:00:00Z",
                source_type: "upload",
                resource_kind: "image",
                has_thumbnail: false,
                preview_url: "/v2/uploads/file_v2_image/preview",
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.listResources({ limit: 50, query: "prairie", kind: "image" });

    expect(response.resources[0].file_id).toBe("file_v2_image");
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/resources?limit=50&offset=0&q=prairie&kind=image",
    ]);
  });

  it("promotes selected resource and dataset URIs into the V2 run envelope", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url.endsWith("/v2/threads")) {
        return new Response(JSON.stringify({ thread_id: "thread_v2_123" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith("/v2/threads/thread_v2_123/runs")) {
        return new Response(JSON.stringify({ run_id: "run_v2_123" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/v2/runs/run_v2_123/events")) {
        return new Response(
          encoder.encode(
            'event: run_event\ndata: {"run_id":"run_v2_123","thread_id":"thread_v2_123","event_kind":"run.completed","payload":{"response_text":"done"}}\n\n'
          ),
          { status: 200, headers: { "Content-Type": "text/event-stream" } }
        );
      }
      if (url.endsWith("/v2/runs/run_v2_123")) {
        return new Response(JSON.stringify({ run_id: "run_v2_123", response_text: "done" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    await client.chatStream({
      messages: [{ role: "user", content: "run prairie dog detection" }],
      uploaded_files: [],
      file_ids: ["file-local"],
      conversation_id: "conversation-local-123",
      goal: "run prairie dog detection",
      selection_context: {
        resource_uris: ["bisque://resource/1"],
        dataset_uris: ["bisque://dataset/2"],
      },
    });

    const runCreateCall = fetchMock.mock.calls.find(([input]) =>
      String(input).endsWith("/v2/threads/thread_v2_123/runs")
    );
    if (!runCreateCall) {
      throw new Error("expected V2 run creation call");
    }
    const [, init] = runCreateCall;
    const body = JSON.parse(String(init?.body));
    expect(body).toMatchObject({
      file_ids: ["file-local"],
      resource_uris: ["bisque://resource/1"],
      dataset_uris: ["bisque://dataset/2"],
    });
  });

  it("recovers when a cached V2 thread id was lost by the backend", async () => {
    browserStorage().setItem(
      "bisque-ultra:v2-chat-thread-map",
      JSON.stringify({ "conversation-local-123": "thread_stale" })
    );

    const encoder = new TextEncoder();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url.endsWith("/v2/threads/thread_stale")) {
        return new Response("404 page not found", { status: 404 });
      }
      if (url.endsWith("/v2/threads")) {
        return new Response(JSON.stringify({ thread_id: "thread_fresh" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith("/v2/threads/thread_fresh/runs")) {
        return new Response(JSON.stringify({ run_id: "run_fresh" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/v2/runs/run_fresh/events")) {
        return new Response(
          encoder.encode(
            'event: run_event\ndata: {"run_id":"run_fresh","thread_id":"thread_fresh","event_kind":"run.completed","payload":{"response_text":"done"}}\n\n'
          ),
          { status: 200, headers: { "Content-Type": "text/event-stream" } }
        );
      }
      if (url.endsWith("/v2/runs/run_fresh")) {
        return new Response(JSON.stringify({ run_id: "run_fresh", response_text: "done" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.chatStream({
      messages: [{ role: "user", content: "create a plot" }],
      uploaded_files: [],
      conversation_id: "conversation-local-123",
      goal: "create a plot",
    });

    expect(response.run_id).toBe("run_fresh");
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads/thread_stale",
      "https://ultra.example.org/v2/threads",
      "https://ultra.example.org/v2/threads/thread_fresh/runs",
      "https://ultra.example.org/v2/runs/run_fresh/events?stream=true&after_sequence=0",
      "https://ultra.example.org/v2/runs/run_fresh",
    ]);
    expect(JSON.parse(browserStorage().getItem("bisque-ultra:v2-chat-thread-map") ?? "{}")).toEqual({
      "conversation-local-123": "thread_fresh",
    });
  });

  it("retries run creation once when a cached thread disappears after validation", async () => {
    browserStorage().setItem(
      "bisque-ultra:v2-chat-thread-map",
      JSON.stringify({ "conversation-local-123": "thread_racy" })
    );

    const encoder = new TextEncoder();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url.endsWith("/v2/threads/thread_racy")) {
        return new Response(JSON.stringify({ thread_id: "thread_racy" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith("/v2/threads/thread_racy/runs")) {
        return new Response("404 page not found", { status: 404 });
      }
      if (url.endsWith("/v2/threads")) {
        return new Response(JSON.stringify({ thread_id: "thread_recovered" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith("/v2/threads/thread_recovered/runs")) {
        return new Response(JSON.stringify({ run_id: "run_recovered" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/v2/runs/run_recovered/events")) {
        return new Response(
          encoder.encode(
            'event: run_event\ndata: {"run_id":"run_recovered","thread_id":"thread_recovered","event_kind":"run.completed","payload":{"response_text":"done"}}\n\n'
          ),
          { status: 200, headers: { "Content-Type": "text/event-stream" } }
        );
      }
      if (url.endsWith("/v2/runs/run_recovered")) {
        return new Response(JSON.stringify({ run_id: "run_recovered", response_text: "done" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.chatStream({
      messages: [{ role: "user", content: "create a plot" }],
      uploaded_files: [],
      conversation_id: "conversation-local-123",
      goal: "create a plot",
      idempotency_key: "message-key-123",
    });

    expect(response.run_id).toBe("run_recovered");
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads/thread_racy",
      "https://ultra.example.org/v2/threads/thread_racy/runs",
      "https://ultra.example.org/v2/threads",
      "https://ultra.example.org/v2/threads/thread_recovered/runs",
      "https://ultra.example.org/v2/runs/run_recovered/events?stream=true&after_sequence=0",
      "https://ultra.example.org/v2/runs/run_recovered",
    ]);
    expect(JSON.parse(browserStorage().getItem("bisque-ultra:v2-chat-thread-map") ?? "{}")).toEqual({
      "conversation-local-123": "thread_recovered",
    });
  });

  it("lists conversations from V2 threads without probing legacy conversation routes", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/threads?limit=25&offset=0") {
        return new Response(
          JSON.stringify({
            count: 1,
            threads: [
              {
                thread_id: "thread_v2_123",
                title: "create a matplotlib y",
                status: "active",
                created_at: "2026-05-31T11:16:00Z",
                updated_at: "2026-05-31T11:17:00Z",
                latest_run_id: "run_v2_123",
                metadata: {
                  conversation_id: "conversation-local-123",
                  preview: "create a matplotlib y = x^2 plot",
                  message_count: 2,
                  frontend_state: {
                    preferredPanel: "chat",
                    messages: [{ id: "msg-1", role: "user", content: "hello" }],
                  },
                },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.listConversations(25, 0, true);

    expect(response).toMatchObject({
      count: 1,
      total_count: 1,
      limit: 25,
      offset: 0,
      has_more: false,
      conversations: [
        {
          conversation_id: "conversation-local-123",
          title: "create a matplotlib y",
          preview: "create a matplotlib y = x^2 plot",
          message_count: 2,
          preferred_panel: "chat",
          running: false,
          state: {
            preferredPanel: "chat",
            messages: [{ id: "msg-1", role: "user", content: "hello" }],
          },
        },
      ],
    });
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads?limit=25&offset=0",
    ]);
  });

  it("persists conversation snapshots through V2 threads without probing legacy upsert routes", async () => {
    browserStorage().setItem(
      "bisque-ultra:v2-chat-thread-map",
      JSON.stringify({ "conversation-local-123": "thread_v2_123" })
    );
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/threads/thread_v2_123") {
        return new Response(
          JSON.stringify({
            thread_id: "thread_v2_123",
            title: "create a matplotlib y",
            status: "active",
            created_at: "2026-05-31T11:16:00Z",
            updated_at: "2026-05-31T11:17:00Z",
            metadata: {
              conversation_id: "conversation-local-123",
              preview: "create a matplotlib y = x^2 plot",
              message_count: 2,
              frontend_state: {
                messages: [{ id: "msg-1", role: "user", content: "hello" }],
              },
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.upsertConversation({
      conversation_id: "conversation-local-123",
      title: "create a matplotlib y",
      created_at_ms: Date.parse("2026-05-31T11:16:00Z"),
      updated_at_ms: Date.parse("2026-05-31T11:17:00Z"),
      preview: "create a matplotlib y = x^2 plot",
      message_count: 2,
      preferred_panel: "chat",
      running: false,
      state: {
        messages: [{ id: "msg-1", role: "user", content: "hello" }],
      },
    });

    expect(response.conversation_id).toBe("conversation-local-123");
    const v2UpsertCall = fetchMock.mock.calls.find(([input]) =>
      String(input).endsWith("/v2/threads/thread_v2_123")
    );
    expect(v2UpsertCall).toBeTruthy();
    const [, init] = v2UpsertCall ?? [];
    expect(init?.method).toBe("PUT");
    expect(JSON.parse(String(init?.body))).toMatchObject({
      title: "create a matplotlib y",
      metadata: {
        conversation_id: "conversation-local-123",
        preview: "create a matplotlib y = x^2 plot",
        message_count: 2,
        frontend_state: {
          messages: [{ id: "msg-1", role: "user", content: "hello" }],
        },
      },
    });
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads/thread_v2_123",
    ]);
  });

  it("persists terminal chat state through V2 thread metadata", async () => {
    browserStorage().setItem(
      "bisque-ultra:v2-chat-thread-map",
      JSON.stringify({ "conversation-local-123": "thread_v2_123" })
    );
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/threads/thread_v2_123") {
        return new Response(
          JSON.stringify({
            thread_id: "thread_v2_123",
            title: "write code and visualize",
            status: "active",
            created_at: "2026-05-31T15:04:00Z",
            updated_at: "2026-05-31T15:05:00Z",
            latest_run_id: "run_v2_123",
            metadata: {
              conversation_id: "conversation-local-123",
              preview: "Write code and visualize how bubble sort works",
              message_count: 2,
              frontend_state: {
                sending: false,
                streamingMessageId: null,
                messages: [
                  {
                    id: "msg-user",
                    role: "user",
                    content: "Write code and visualize how bubble sort works",
                  },
                  {
                    id: "msg-assistant",
                    role: "assistant",
                    runId: "run_v2_123",
                    content: "Bubble sort explanation",
                  },
                ],
              },
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.upsertConversation({
      conversation_id: "conversation-local-123",
      title: "write code and visualize",
      created_at_ms: Date.parse("2026-05-31T15:04:00Z"),
      updated_at_ms: Date.parse("2026-05-31T15:05:00Z"),
      preview: "Write code and visualize how bubble sort works",
      message_count: 2,
      preferred_panel: "chat",
      running: false,
      state: {
        sending: false,
        streamingMessageId: null,
        messages: [
          {
            id: "msg-user",
            role: "user",
            content: "Write code and visualize how bubble sort works",
          },
          {
            id: "msg-assistant",
            role: "assistant",
            runId: "run_v2_123",
            content: "Bubble sort explanation",
          },
        ],
      },
    });

    expect(response.conversation_id).toBe("conversation-local-123");
    const v2UpsertCall = fetchMock.mock.calls.find(([input]) =>
      String(input).endsWith("/v2/threads/thread_v2_123")
    );
    expect(v2UpsertCall).toBeTruthy();
    const [, init] = v2UpsertCall ?? [];
    expect(init?.method).toBe("PUT");
    expect(JSON.parse(String(init?.body))).toMatchObject({
      metadata: {
        conversation_id: "conversation-local-123",
        frontend_state: {
          sending: false,
          streamingMessageId: null,
          messages: [
            {
              role: "user",
              content: "Write code and visualize how bubble sort works",
            },
            {
              role: "assistant",
              runId: "run_v2_123",
              content: "Bubble sort explanation",
            },
          ],
        },
      },
    });
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads/thread_v2_123",
    ]);
  });

  it("hydrates a V2-only conversation from thread messages and latest run without probing legacy routes", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/threads?limit=1000&offset=0") {
        return new Response(
          JSON.stringify({
            count: 1,
            threads: [
              {
                thread_id: "thread_v2_123",
                title: "create a matplotlib y",
                status: "active",
                created_at: "2026-05-31T11:16:00Z",
                updated_at: "2026-05-31T11:17:00Z",
                latest_run_id: "run_v2_123",
                metadata: {
                  conversation_id: "conversation-local-123",
                },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/threads/thread_v2_123/messages") {
        return new Response(
          JSON.stringify({
            thread_id: "thread_v2_123",
            count: 1,
            messages: [
              {
                message_id: "msg_user_1",
                role: "user",
                content: "create a matplotlib y = x^2 plot",
                created_at: "2026-05-31T11:16:00Z",
                run_id: "run_v2_123",
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/runs/run_v2_123") {
        return new Response(
          JSON.stringify({
            run_id: "run_v2_123",
            thread_id: "thread_v2_123",
            status: "succeeded",
            goal: "create a matplotlib y = x^2 plot",
            response_text: "The plot demonstrates quadratic growth.",
            created_at: "2026-05-31T11:16:00Z",
            updated_at: "2026-05-31T11:17:00Z",
            completed_at: "2026-05-31T11:17:00Z",
            metadata: {
              response_layout: { sections: [] },
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.getConversation("conversation-local-123");

    expect(response.state.messages).toEqual([
      {
        id: "msg_user_1",
        role: "user",
        content: "create a matplotlib y = x^2 plot",
        createdAt: Date.parse("2026-05-31T11:16:00Z"),
        runId: "run_v2_123",
      },
      {
        id: "run_v2_123-assistant",
        role: "assistant",
        content: "The plot demonstrates quadratic growth.",
        createdAt: Date.parse("2026-05-31T11:17:00Z"),
        runId: "run_v2_123",
        responseMetadata: {
          response_layout: { sections: [] },
        },
      },
    ]);
    expect(response.preview).toBe("create a matplotlib y = x^2 plot");
    expect(response.message_count).toBe(2);
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads?limit=1000&offset=0",
      "https://ultra.example.org/v2/threads/thread_v2_123/messages",
      "https://ultra.example.org/v2/runs/run_v2_123",
    ]);
  });

  it("hydrates a V2-only running latest run with an assistant placeholder for stream recovery", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/threads?limit=1000&offset=0") {
        return new Response(
          JSON.stringify({
            count: 1,
            threads: [
              {
                thread_id: "thread_v2_123",
                title: "create a matplotlib y",
                status: "active",
                created_at: "2026-05-31T11:16:00Z",
                updated_at: "2026-05-31T11:17:00Z",
                latest_run_id: "run_v2_123",
                metadata: {
                  conversation_id: "conversation-local-123",
                },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/threads/thread_v2_123/messages") {
        return new Response(
          JSON.stringify({
            thread_id: "thread_v2_123",
            count: 1,
            messages: [
              {
                message_id: "msg_user_1",
                role: "user",
                content: "create a matplotlib y = x^2 plot",
                created_at: "2026-05-31T11:16:00Z",
                run_id: "run_v2_123",
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/runs/run_v2_123") {
        return new Response(
          JSON.stringify({
            run_id: "run_v2_123",
            thread_id: "thread_v2_123",
            status: "running",
            goal: "create a matplotlib y = x^2 plot",
            response_text: "",
            created_at: "2026-05-31T11:16:00Z",
            updated_at: "2026-05-31T11:17:00Z",
            metadata: {},
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.getConversation("conversation-local-123");

    expect(response.state.messages).toEqual([
      {
        id: "msg_user_1",
        role: "user",
        content: "create a matplotlib y = x^2 plot",
        createdAt: Date.parse("2026-05-31T11:16:00Z"),
        runId: "run_v2_123",
      },
      {
        id: "run_v2_123-assistant",
        role: "assistant",
        content: "",
        createdAt: Date.parse("2026-05-31T11:17:00Z"),
        runId: "run_v2_123",
        responseMetadata: {},
      },
    ]);
    expect(response.message_count).toBe(2);
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads?limit=1000&offset=0",
      "https://ultra.example.org/v2/threads/thread_v2_123/messages",
      "https://ultra.example.org/v2/runs/run_v2_123",
    ]);
  });

  it("searches conversations from V2 threads without probing legacy search routes", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/threads?limit=50&offset=0") {
        return new Response(
          JSON.stringify({
            count: 2,
            threads: [
              {
                thread_id: "thread_pca",
                title: "Iris PCA analysis",
                created_at: "2026-05-31T11:16:00Z",
                updated_at: "2026-05-31T11:17:00Z",
                metadata: {
                  conversation_id: "conversation-pca",
                  preview: "Create a PCA plot and table",
                },
              },
              {
                thread_id: "thread_sort",
                title: "Bubble sort visualization",
                created_at: "2026-05-31T11:18:00Z",
                updated_at: "2026-05-31T11:19:00Z",
                metadata: {
                  conversation_id: "conversation-sort",
                  preview: "Show how bubble sort works",
                },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.searchConversations("pca", 50);

    expect(response).toMatchObject({
      query: "pca",
      count: 1,
      matches: [
        {
          conversation_id: "conversation-pca",
          title: "Iris PCA analysis",
        },
      ],
    });
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads?limit=50&offset=0",
    ]);
  });

  it("gets run results from V2 run records without probing legacy run result routes", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/runs/run_v2_123") {
        return new Response(
          JSON.stringify({
            run_id: "run_v2_123",
            thread_id: "thread_v2_123",
            goal: "create a matplotlib y = x^2 plot",
            status: "succeeded",
            workflow_kind: "deep_agents",
            response_text: "The plot demonstrates quadratic growth.",
            created_at: "2026-05-31T11:16:00Z",
            updated_at: "2026-05-31T11:17:00Z",
            metadata: {
              response_layout: { sections: [] },
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.getRunResult("run_v2_123");

    expect(response).toEqual({
      run_id: "run_v2_123",
      status: "succeeded",
      result: {
        run_id: "run_v2_123",
        model: "deep_agents",
        response_text: "The plot demonstrates quadratic growth.",
        duration_seconds: 0,
        progress_events: [],
        benchmark: null,
        metadata: {
          response_layout: { sections: [] },
        },
      },
    });
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/runs/run_v2_123",
    ]);
  });

  it("deletes conversations through V2 threads without probing legacy delete routes", async () => {
    browserStorage().setItem(
      "bisque-ultra:v2-chat-thread-map",
      JSON.stringify({ "conversation-local-123": "thread_v2_123" })
    );
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/threads/thread_v2_123") {
        expect(init?.method).toBe("DELETE");
        return new Response(null, { status: 204 });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    await expect(client.deleteConversation("conversation-local-123")).resolves.toEqual({
      deleted: true,
      conversation_id: "conversation-local-123",
    });
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads/thread_v2_123",
    ]);
    expect(JSON.parse(browserStorage().getItem("bisque-ultra:v2-chat-thread-map") ?? "{}")).toEqual({});
  });

  it("uses V2 health, config, and local auth endpoints", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/health") {
        return new Response(JSON.stringify({ status: "ok", ts: "2026-05-31T00:00:00Z" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/config/public") {
        return new Response(JSON.stringify({ app_name: "BisQue Ultra", features: {} }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/auth/session") {
        return new Response(JSON.stringify({ authenticated: true, user: { id: "local-user" } }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/auth/guest") {
        return new Response(JSON.stringify({ authenticated: true, mode: "guest" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/auth/login") {
        return new Response(JSON.stringify({ authenticated: true, mode: "bisque" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/auth/logout") {
        return new Response(JSON.stringify({ authenticated: false, user: null }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    await client.health();
    await client.getPublicConfig();
    await client.getBisqueSession();
    await client.continueAsGuest({ name: "Grace", email: "", affiliation: "" });
    await client.loginBisque({ username: "local", password: "local" });
    await client.logoutBisque();

    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls.every((url) => url.includes("/v2/"))).toBe(true);
    expect(urls.some((url) => url.includes("/v1/"))).toBe(false);
  });

  it("uses V2 run and artifact recovery endpoints without legacy fallbacks", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/runs/run-1") {
        return new Response(
          JSON.stringify({
            run_id: "run-1",
            goal: "plot",
            status: "succeeded",
            created_at: "2026-05-31T00:00:00Z",
            updated_at: "2026-05-31T00:00:01Z",
            workflow_kind: "deep_agents",
            mode: "durable",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/runs/missing/events?limit=2&after_sequence=0") {
        return new Response(JSON.stringify({ error: "not found" }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/runs/missing/artifacts?limit=2") {
        return new Response(JSON.stringify({ error: "not found" }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    await expect(client.getRun("run-1")).resolves.toMatchObject({ run_id: "run-1" });
    await expect(client.getRunEvents("missing", 2)).rejects.toMatchObject({ status: 404 });
    await expect(client.listArtifacts("missing", 2)).rejects.toMatchObject({ status: 404 });
    expect(client.artifactDownloadUrl("run-1", "reports/output.json")).toBe(
      "https://ultra.example.org/v2/runs/run-1/artifacts/download?path=reports%2Foutput.json"
    );

    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls.some((url) => url.includes("/v1/"))).toBe(false);
  });

  it("does not expose legacy V3 chat-session endpoints from the active API client", () => {
    const source = readFileSync(path.join(process.cwd(), "src/lib/api.ts"), "utf8");

    expect(source).not.toContain("/v3/");
    expect(source).not.toMatch(/\b(create|list|get|resolve)V3/);
  });

  it("loads upload viewer metadata from V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/uploads/file-1/viewer") {
        return new Response(
          JSON.stringify({
            kind: "image",
            file_id: "file-1",
            original_name: "prairie.png",
            axis_sizes: { T: 1, C: 3, Z: 1, Y: 2, X: 3 },
            selected_indices: { T: 0, C: 0, Z: 0 },
            service_urls: {
              preview: "/v2/uploads/file-1/preview",
              display: "/v2/uploads/file-1/display",
            },
            metadata: {},
            viewer: {},
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const viewer = await client.getUploadViewer("file-1");

    expect(viewer.file_id).toBe("file-1");
    expect(viewer.axis_sizes.X).toBe(3);
    expect(viewer.service_urls?.display).toBe("/v2/uploads/file-1/display");
    expect(viewer.service_urls?.slice).toBe("/v2/uploads/file-1/slice");
    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls).toEqual(["https://ultra.example.org/v2/uploads/file-1/viewer"]);
    expect(urls.some((url) => url.includes("/v1/"))).toBe(false);
  });

  it("loads upload captions from V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/uploads/file-1/caption") {
        return new Response(
          JSON.stringify({
            file_id: "file-1",
            caption: "Uploaded image prairie.png.",
            source: "fallback",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const caption = await client.getUploadCaption("file-1");

    expect(caption.caption).toContain("prairie.png");
    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls).toEqual(["https://ultra.example.org/v2/uploads/file-1/caption"]);
    expect(urls.some((url) => url.includes("/v1/"))).toBe(false);
  });

  it("posts interactive SAM3 segmentation requests to V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/segment/sam3/interactive") {
        return new Response(
          JSON.stringify({
            success: true,
            run_id: "run_sam3_test",
            response_text: "segmentation accepted",
            progress_events: [],
            result: {
              processed: 0,
              total_files: 1,
              total_masks_generated: 0,
              files_processed: [],
              preferred_upload_paths: [],
              visualization_paths: [],
              output_directories: [],
              annotations: [],
              run_id: "run_sam3_test",
            },
            warnings: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.sam3InteractiveSegment({
      file_ids: ["file-1"],
      annotations: [{ file_id: "file-1", points: [], boxes: [] }],
    });

    expect(response.run_id).toBe("run_sam3_test");
    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls).toEqual(["https://ultra.example.org/v2/segment/sam3/interactive"]);
    expect(urls.some((url) => url.includes("/v1/"))).toBe(false);
  });

  it("posts BisQue resource imports to V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/uploads/from-bisque") {
        return new Response(
          JSON.stringify({
            file_count: 0,
            uploaded: [],
            imports: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.importBisqueResources(["https://bisque.example.org/data_service/image/1"]);

    expect(response.file_count).toBe(0);
    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls).toEqual(["https://ultra.example.org/v2/uploads/from-bisque"]);
    expect(urls.some((url) => url.includes("/v1/"))).toBe(false);
  });

  it("loads admin read models from V2 instead of legacy admin routes", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/admin/overview?top_users=8&issue_limit=12") {
        return new Response(
          JSON.stringify({
            generated_at: "2026-05-31T00:00:00Z",
            kpis: {},
            usage_last_24h: [],
            tool_usage_7d: [],
            top_users: [],
            recent_issues: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/admin/orgs?limit=25&q=allen") {
        return new Response(JSON.stringify({ count: 0, organizations: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/admin/users?limit=25&q=ada") {
        return new Response(JSON.stringify({ count: 0, users: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/admin/runs?limit=10&offset=0&status=running") {
        return new Response(JSON.stringify({ count: 0, runs: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/admin/issues?limit=5") {
        return new Response(JSON.stringify({ count: 0, issues: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    await client.getAdminOverview({ topUsers: 8, issueLimit: 12 });
    await client.listAdminOrganizations({ limit: 25, query: "allen" });
    await client.listAdminUsers({ limit: 25, query: "ada" });
    await client.listAdminRuns({ limit: 10, status: "running" });
    await client.listAdminIssues(5);

    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls.every((url) => url.includes("/v2/admin/"))).toBe(true);
    expect(urls.some((url) => url.includes("/v1/admin/"))).toBe(false);
  });

  it("creates admin organizations through V2", async () => {
    const urls: string[] = [];
    const bodies: string[] = [];
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        urls.push(url);
        bodies.push(String(init?.body ?? ""));
        if (url === "https://ultra.example.org/v2/admin/orgs") {
          return new Response(
            JSON.stringify({
              org_id: "allen-institute",
              name: "Allen Institute",
              status: "active",
              created_at: "2026-06-01T00:00:00Z",
              updated_at: "2026-06-01T00:00:00Z",
              metadata: {},
            }),
            { status: 201, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("not found", { status: 404 });
      }
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient({
      baseUrl: "https://ultra.example.org",
    });

    const org = await client.createAdminOrganization({
      org_id: "allen-institute",
      name: "Allen Institute",
      status: "active",
    });

    expect(org.org_id).toBe("allen-institute");
    expect(urls).toEqual(["https://ultra.example.org/v2/admin/orgs"]);
    expect(JSON.parse(bodies[0])).toMatchObject({
      org_id: "allen-institute",
      name: "Allen Institute",
      status: "active",
    });
  });

  it("creates admin users through V2", async () => {
    const urls: string[] = [];
    const bodies: string[] = [];
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        urls.push(url);
        bodies.push(String(init?.body ?? ""));
        if (url === "https://ultra.example.org/v2/admin/users") {
          return new Response(
            JSON.stringify({
              user_id: "user_grace",
              email: "grace@example.org",
              display_name: "Grace Hopper",
              role: "admin",
              status: "active",
              org_id: "local-org",
              created_at: "2026-06-01T00:00:00Z",
              updated_at: "2026-06-01T00:00:00Z",
              metadata: {},
            }),
            { status: 201, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("not found", { status: 404 });
      }
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient({
      baseUrl: "https://ultra.example.org",
    });

    const user = await client.createAdminUser({
      email: "grace@example.org",
      display_name: "Grace Hopper",
      role: "admin",
      org_id: "local-org",
    });

    expect(user.user_id).toBe("user_grace");
    expect(urls).toEqual(["https://ultra.example.org/v2/admin/users"]);
    expect(JSON.parse(bodies[0])).toMatchObject({
      email: "grace@example.org",
      display_name: "Grace Hopper",
      role: "admin",
      org_id: "local-org",
    });
  });

  it("soft-removes admin users through V2", async () => {
    const urls: string[] = [];
    const methods: string[] = [];
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        urls.push(url);
        methods.push(String(init?.method ?? "GET"));
        if (url === "https://ultra.example.org/v2/admin/users/user_grace") {
          return new Response(
            JSON.stringify({
              user_id: "user_grace",
              email: "grace@example.org",
              display_name: "Grace Hopper",
              role: "admin",
              status: "disabled",
              org_id: "local-org",
              created_at: "2026-06-01T00:00:00Z",
              updated_at: "2026-06-01T00:01:00Z",
              metadata: {},
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("not found", { status: 404 });
      }
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient({
      baseUrl: "https://ultra.example.org",
    });

    const user = await client.deleteAdminUser("user_grace");

    expect(user.status).toBe("disabled");
    expect(urls).toEqual(["https://ultra.example.org/v2/admin/users/user_grace"]);
    expect(methods).toEqual(["DELETE"]);
  });

  it("requeues admin runs through V2 with an explicit reason", async () => {
    const urls: string[] = [];
    const bodies: string[] = [];
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        urls.push(url);
        bodies.push(String(init?.body ?? ""));
        if (url === "https://ultra.example.org/v2/admin/runs/run_1/requeue") {
          return new Response(
            JSON.stringify({
              run_id: "run_1",
              previous_status: "running",
              status: "running",
              updated: true,
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("not found", { status: 404 });
      }
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient({
      baseUrl: "https://ultra.example.org",
    });

    const action = await client.requeueAdminRun("run_1", "expired lease");

    expect(action.updated).toBe(true);
    expect(urls).toEqual(["https://ultra.example.org/v2/admin/runs/run_1/requeue"]);
    expect(JSON.parse(bodies[0])).toEqual({ reason: "expired lease" });
  });

  it("loads training read models from V2 instead of legacy training routes", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/training/models") {
        return new Response(JSON.stringify({ count: 0, models: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/training/prairie/status") {
        return new Response(
          JSON.stringify({
            dataset_name: "Prairie Active Learning",
            model_health: "Watch",
            reviewed_images: 0,
            unreviewed_images: 0,
            class_counts: {},
            unsupported_class_counts: {},
            detection_counts: {},
            latest_metrics: {},
            benchmark_baseline: {},
            benchmark_latest_candidate: {},
            benchmark_ready: false,
            canonical_benchmark_ready: false,
            promotion_benchmark_ready: false,
            retrain_gate: false,
            retrain_gate_reasons: [],
            retrain_gate_counts: {},
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/training/prairie/retrain-requests") {
        return new Response(JSON.stringify({ count: 0, requests: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/training/domains?limit=200") {
        return new Response(JSON.stringify({ count: 0, domains: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/training/domains/prairie/lineages?limit=50") {
        return new Response(JSON.stringify({ count: 0, lineages: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "https://ultra.example.org/v2/training/lineages/lineage-1/versions?limit=25") {
        return new Response(JSON.stringify({ count: 0, versions: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    await client.listTrainingModels();
    await client.getPrairieActiveLearningStatus();
    await client.listPrairieRetrainRequests();
    await client.listTrainingDomains(200);
    await client.listDomainLineages("prairie", { limit: 50 });
    await client.listLineageVersions("lineage-1", { limit: 25 });

    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls.every((url) => url.includes("/v2/training/"))).toBe(true);
    expect(urls.some((url) => url.includes("/v1/training/"))).toBe(false);
  });

  it("paginates V2 run events so long autonomous traces hydrate completely", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/runs/run_long/events?limit=2&after_sequence=0") {
        return new Response(
          JSON.stringify({
            run_id: "run_long",
            count: 2,
            events: [
              {
                event_id: "evt_1",
                sequence: 1,
                run_id: "run_long",
                event_kind: "run.started",
                payload: {},
              },
              {
                event_id: "evt_2",
                sequence: 2,
                run_id: "run_long",
                event_kind: "tool_call.started",
                payload: { tool_name: "execute" },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/runs/run_long/events?limit=2&after_sequence=2") {
        return new Response(
          JSON.stringify({
            run_id: "run_long",
            count: 1,
            events: [
              {
                event_id: "evt_3",
                sequence: 3,
                run_id: "run_long",
                event_kind: "run.completed",
                payload: { response_text: "done" },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.getRunEvents("run_long", 2);

    expect(response.events.map((event) => event.event_type)).toEqual([
      "run.started",
      "tool_call.started",
      "run.completed",
    ]);
    expect(response.events.map((event) => event.payload?.sequence)).toEqual([1, 2, 3]);
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/runs/run_long/events?limit=2&after_sequence=0",
      "https://ultra.example.org/v2/runs/run_long/events?limit=2&after_sequence=2",
    ]);
  });
});
