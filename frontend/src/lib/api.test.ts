import { readFileSync } from "node:fs";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiClient, UploadPausedError } from "./api";

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
      client.resourceDownloadUrl("file-123"),
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

  it("builds resource download URLs through the scoped V2 resource API", () => {
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    expect(client.resourceDownloadUrl("file/with spaces")).toBe(
      "https://ultra.example.org/v2/resources/file%2Fwith%20spaces/download"
    );
  });

  it("builds uploaded image slice URLs through the V2 upload API", () => {
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    expect(client.uploadSliceUrl("file-123", { axis: "z", z: 2 })).toBe(
      "https://ultra.example.org/v2/uploads/file-123/slice?axis=z&z=2"
    );
  });

  it("can cache-bust transformed uploaded image slice URLs", () => {
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    expect(client.uploadSliceUrl("file-123", { axis: "z", z: 2, cacheKey: "windowed-v1:abc123" })).toBe(
      "https://ultra.example.org/v2/uploads/file-123/slice?axis=z&z=2&cache_key=windowed-v1%3Aabc123"
    );
  });

  it("builds transformed uploaded image display URLs", () => {
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    expect(
      client.uploadDisplayUrl("file-123", "/v2/uploads/file-123/display", {
        enhancement: "hounsfield:1001.500:1.000",
        negative: true,
        gamma: 1.2,
        channels: [1],
        channelColors: ["#00ff00"],
        cacheKey: "windowed-v2:abc123",
      })
    ).toBe(
      "https://ultra.example.org/v2/uploads/file-123/display?enhancement=hounsfield%3A1001.500%3A1.000&negative=true&gamma=1.2&channels=1&channel_colors=%2300ff00&cache_key=windowed-v2%3Aabc123"
    );
  });

  it("builds scientific upload viewer URLs through the V2 upload API", () => {
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    const urls = [
      client.uploadTileUrl("file-123", { axis: "z", level: 0, tileX: 1, tileY: 2, z: 3 }),
      client.uploadAtlasUrl("file-123", { enhancement: "d", t: 1 }),
    ];

    expect(urls).toEqual([
      "https://ultra.example.org/v2/uploads/file-123/tiles/z/0/1/2?z=3",
      "https://ultra.example.org/v2/uploads/file-123/atlas?enhancement=d&t=1",
    ]);
    expect(urls.some((url) => url.includes("/v1/"))).toBe(false);
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
          'event: run_event\ndata: {"run_id":"run_v2_123","thread_id":"thread_v2_123","event_kind":"run.token_usage","level":"debug","message":"","payload":{"usage_event_id":"evt_usage_1","input_tokens":10,"output_tokens":2,"total_tokens":12,"model":"deepseek_v4"}}\n\n',
          'event: run_event\ndata: {"run_id":"run_v2_123","thread_id":"thread_v2_123","event_kind":"message.delta","level":"info","payload":{"delta":"Hello"}}\n\n',
          'event: run_event\ndata: {"run_id":"run_v2_123","thread_id":"thread_v2_123","event_kind":"run.completed","level":"info","payload":{"response_text":"Hello","conversation_title":"Matplotlib Plot Setup","title_generation":{"strategy":"llm","model":"gpt-title"},"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12,"model":"deepseek_v4"}}}\n\n',
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
    expect(response.metadata).toMatchObject({
      conversation_title: "Matplotlib Plot Setup",
      title_generation: {
        strategy: "llm",
        model: "gpt-title",
      },
    });
    expect(tokens).toEqual(["Hello"]);
    expect(runStarts).toEqual(["run_v2_123"]);
    expect(response.metadata?.usage).toEqual({
      input_tokens: 10,
      output_tokens: 2,
      total_tokens: 12,
      model: "deepseek_v4",
    });
    expect(runEvents).toEqual(["run.token_usage", "message.delta", "run.completed"]);
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads",
      "https://ultra.example.org/v2/threads/thread_v2_123/runs",
      "https://ultra.example.org/v2/runs/run_v2_123/events?stream=true&after_sequence=0",
      "https://ultra.example.org/v2/runs/run_v2_123",
    ]);
    const [, threadCreateInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(threadCreateInit.body))).toMatchObject({
      title: "Plot",
      metadata: {
        conversation_id: "conversation-local-123",
        frontend_bridge: "v2-chat",
        title_state: {
          source: "auto",
          strategy: "initial_request",
        },
      },
    });
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

  it("skips duplicate run event sequences so replay overlap never doubles streamed text", async () => {
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
          // A reconnect/replay overlap can re-deliver an already-seen event.
          'event: run_event\ndata: {"run_id":"run_resume","event_kind":"message.delta","sequence":8,"payload":{"text":" more"}}\n\n',
          'event: run_event\ndata: {"run_id":"run_resume","event_kind":"message.delta","sequence":9,"payload":{"text":" text"}}\n\n',
          'event: run_event\ndata: {"run_id":"run_resume","event_kind":"run.completed","sequence":10,"payload":{"response_text":"done more text"}}\n\n',
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
            response_text: "done more text",
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
    const response = await client.resumeRunStream("run_resume", {
      afterSequence: 7,
      onToken: (delta) => tokens.push(delta),
    });

    expect(tokens).toEqual([" more", " text"]);
    expect(response.response_text).toBe("done more text");
  });

  it("fetches run events incrementally from a caller-provided sequence cursor", async () => {
    const urls: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      urls.push(url);
      if (url.includes("after_sequence=500")) {
        return new Response(
          JSON.stringify({
            run_id: "run_inc",
            count: 1,
            events: [
              { event_id: "evt-501", sequence: 501, run_id: "run_inc", event_kind: "message.delta" },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      return new Response("not found", { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    const response = await client.getRunEvents("run_inc", 200, { afterSequence: 500 });

    expect(urls).toEqual([
      "https://ultra.example.org/v2/runs/run_inc/events?limit=200&after_sequence=500",
    ]);
    expect(response.events.map((event) => event.payload?.sequence)).toEqual([501]);
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

  it("does not complete a chat response when the V2 stream ends while the run is still active", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/runs/run_live/events?stream=true&after_sequence=0") {
        const body =
          'event: run_event\ndata: {"run_id":"run_live","event_kind":"message.delta","sequence":1,"payload":{"text":"partial"}}\n\n';
        return new Response(encoder.encode(body), {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      if (url === "https://ultra.example.org/v2/runs/run_live") {
        return new Response(
          JSON.stringify({
            run_id: "run_live",
            status: "running",
            response_text: "",
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
    const donePayloads: unknown[] = [];

    await expect(
      client.resumeRunStream("run_live", {
        onToken: (delta) => tokens.push(delta),
        onDone: (payload) => donePayloads.push(payload),
      })
    ).rejects.toMatchObject({
      status: 503,
      detail: {
        run_id: "run_live",
        status: "running",
      },
    });

    expect(tokens).toEqual(["partial"]);
    expect(donePayloads).toEqual([]);
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/runs/run_live/events?stream=true&after_sequence=0",
      "https://ultra.example.org/v2/runs/run_live",
    ]);
  });

  it("returns only a temporary local title without calling legacy V1 title endpoints", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.chatTitle({
      messages: [{ role: "user", content: "create a matplotlib y = x^2 plot" }],
      max_words: 4,
    });

    expect(response).toEqual({
      title: "Matplotlib Y X 2",
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

  it("loads V2 upload-session status for refresh reconciliation", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toBe("https://ultra.example.org/v2/upload-sessions/upload_session_active");
      expect(init?.method).toBe("GET");
      return new Response(
        JSON.stringify({
          session: {
            session_id: "upload_session_active",
            owner_user_id: "field-user",
            source_type: "upload",
            status: "active",
            total_bytes: 1000,
            bytes_received: 512,
            bytes_verified: 512,
            bytes_committed: 0,
            created_at: "2026-06-08T00:00:00Z",
            updated_at: "2026-06-08T00:00:01Z",
            metadata: {},
          },
          files: [],
          chunks: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const status = await client.getUploadSessionStatus("upload_session_active");

    expect(status.session.bytes_verified).toBe(512);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("pauses and resumes V2 upload sessions through explicit controls", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      expect(init?.method).toBe("POST");
      if (url === "https://ultra.example.org/v2/upload-sessions/upload_session_field/pause") {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_field",
              owner_user_id: "field-user",
              source_type: "upload",
              status: "paused",
              total_bytes: 1000,
              bytes_received: 512,
              bytes_verified: 512,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
              metadata: {},
            },
            files: [],
            chunks: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/upload-sessions/upload_session_field/resume") {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_field",
              owner_user_id: "field-user",
              source_type: "upload",
              status: "active",
              total_bytes: 1000,
              bytes_received: 512,
              bytes_verified: 512,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
              metadata: {},
            },
            files: [],
            chunks: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`unexpected upload-session control URL ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const paused = await client.pauseUploadSession("upload_session_field");
    const resumed = await client.resumeUploadSession("upload_session_field");

    expect(paused.session.status).toBe("paused");
    expect(resumed.session.status).toBe("active");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("stops scheduling chunks when an upload session is paused locally", async () => {
    const chunkSize = 8 * 1024 * 1024;
    const firstChunk = new Uint8Array(chunkSize);
    firstChunk.fill(7);
    const tail = new Uint8Array([8, 9, 10]);
    const file = new File([firstChunk, tail], "paused-field-volume.nii", {
      type: "application/x-nifti",
      lastModified: 1_780_915_200_000,
    });

    let fileToken = "";
    let paused = false;
    const urls: string[] = [];
    const progressEvents: Array<{ status: string; bytesVerified: number }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      urls.push(url);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body)) as {
          files: Array<{ file_token: string; original_name: string; size_bytes: number }>;
        };
        fileToken = body.files[0].file_token;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_pause_local",
              owner_user_id: "field-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_pause_local",
                file_token: fileToken,
                original_name: "paused-field-volume.nii",
                content_type: "application/x-nifti",
                size_bytes: file.size,
                status: "pending",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:00Z",
              },
            ],
            chunks: [],
            limits: {
              max_parallel_files: 1,
              max_parallel_chunks: 1,
              max_files_per_session: 1000,
            },
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${fileToken}/chunks/0`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_pause_local",
              owner_user_id: "field-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: chunkSize,
              bytes_verified: chunkSize,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_pause_local",
              file_token: fileToken,
              original_name: "paused-field-volume.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
            },
            chunk: {
              session_id: "upload_session_pause_local",
              file_token: fileToken,
              chunk_index: 0,
              offset: 0,
              size_bytes: chunkSize,
              sha256: new Headers(init?.headers).get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.includes("/chunks/1")) {
        throw new Error("paused upload should not schedule the next chunk");
      }
      if (url.endsWith(`/files/${fileToken}/complete`)) {
        throw new Error("paused upload should not commit the file");
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    await expect(
      client.uploadFiles([file], {
        pauseSignal: {
          isPaused: () => paused,
        },
        onProgress: (event) => {
          progressEvents.push({ status: event.status, bytesVerified: event.bytesVerified });
          if (event.status === "uploading" && event.bytesVerified >= chunkSize) {
            paused = true;
          }
        },
      })
    ).rejects.toBeInstanceOf(UploadPausedError);

    expect(urls.some((url) => url.includes("/chunks/1"))).toBe(false);
    expect(urls.some((url) => url.endsWith(`/files/${fileToken}/complete`))).toBe(false);
    expect(progressEvents).toContainEqual({ status: "paused", bytesVerified: chunkSize });
  });

  it("cancels V2 upload sessions through an explicit control", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toBe("https://ultra.example.org/v2/upload-sessions/upload_session_field/cancel");
      expect(init?.method).toBe("POST");
      return new Response(
        JSON.stringify({
          session: {
            session_id: "upload_session_field",
            owner_user_id: "field-user",
            source_type: "upload",
            status: "canceled",
            total_bytes: 1000,
            bytes_received: 512,
            bytes_verified: 512,
            bytes_committed: 0,
            error: "canceled by user",
            created_at: "2026-06-08T00:00:00Z",
            updated_at: "2026-06-08T00:00:04Z",
            metadata: {},
          },
          files: [],
          chunks: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const canceled = await client.cancelUploadSession("upload_session_field");

    expect(canceled.session.status).toBe("canceled");
    expect(canceled.session.error).toBe("canceled by user");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("uploads local chat files through V2 upload sessions without probing legacy upload routes", async () => {
    let createdFileToken = "";
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        expect(init?.method).toBe("POST");
        const body = JSON.parse(String(init?.body));
        expect(body).toMatchObject({
          total_bytes: 4,
          files: [
            {
              original_name: "prairie.jpg",
              content_type: "image/jpeg",
              size_bytes: 4,
            },
          ],
        });
        createdFileToken = body.files[0].file_token;
        expect(createdFileToken).toMatch(/^file-0-/);
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_v2_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 4,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_v2_1",
                file_token: createdFileToken,
                original_name: "prairie.jpg",
                content_type: "image/jpeg",
                size_bytes: 4,
                status: "pending",
                created_at: "2026-05-31T00:00:00Z",
                updated_at: "2026-05-31T00:00:00Z",
              },
            ],
            chunks: [],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      if (
        createdFileToken &&
        url ===
          `https://ultra.example.org/v2/upload-sessions/upload_session_v2_1/files/${createdFileToken}/chunks/0`
      ) {
        expect(init?.method).toBe("PUT");
        const headers = new Headers(init?.headers);
        expect(headers.get("content-type")).toBe("application/octet-stream");
        expect(headers.get("x-upload-offset")).toBe("0");
        expect(headers.get("x-upload-chunk-sha256")).toMatch(/^[a-f0-9]{64}$/);
        expect(init?.body).toBeInstanceOf(Blob);
        expect(await (init?.body as Blob).text()).toBe("data");
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_v2_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 4,
              bytes_received: 4,
              bytes_verified: 4,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_v2_1",
              file_token: createdFileToken,
              original_name: "prairie.jpg",
              content_type: "image/jpeg",
              size_bytes: 4,
              status: "uploading",
              created_at: "2026-05-31T00:00:00Z",
              updated_at: "2026-05-31T00:00:01Z",
            },
            chunk: {
              session_id: "upload_session_v2_1",
              file_token: createdFileToken,
              chunk_index: 0,
              offset: 0,
              size_bytes: 4,
              sha256: headers.get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (
        createdFileToken &&
        url ===
          `https://ultra.example.org/v2/upload-sessions/upload_session_v2_1/files/${createdFileToken}/complete`
      ) {
        expect(init?.method).toBe("POST");
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_v2_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "completed",
              total_bytes: 4,
              bytes_received: 4,
              bytes_verified: 4,
              bytes_committed: 4,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_v2_1",
              file_token: createdFileToken,
              resource_id: "file_v2_image",
              original_name: "prairie.jpg",
              content_type: "image/jpeg",
              size_bytes: 4,
              computed_sha256: "abc123",
              status: "completed",
              created_at: "2026-05-31T00:00:00Z",
              updated_at: "2026-05-31T00:00:02Z",
            },
            resource: {
              file_id: "file_v2_image",
              original_name: "prairie.jpg",
              content_type: "image/jpeg",
              size_bytes: 4,
              sha256: "abc123",
              created_at: "2026-05-31T00:00:00Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const progressEvents: Array<{
      status: string;
      bytesVerified: number;
      fingerprint?: string;
      contentType?: string;
      chunkSizeBytes?: number;
      fileToken?: string;
      sessionId?: string;
    }> = [];
    const file = new File(["data"], "prairie.jpg", {
      type: "image/jpeg",
      lastModified: 1_780_915_200_000,
    });
    const response = await client.uploadFiles([
      file,
    ], {
      onProgress: (event) => {
        progressEvents.push({
          status: event.status,
          bytesVerified: event.bytesVerified,
          fingerprint: event.fingerprint,
          contentType: event.contentType,
          chunkSizeBytes: event.chunkSizeBytes,
          fileToken: event.fileToken,
          sessionId: event.sessionId,
        });
      },
    });

    expect(response.uploaded[0].file_id).toBe("file_v2_image");
    expect(progressEvents).toMatchObject([
      {
        status: "creating",
        bytesVerified: 0,
        contentType: "image/jpeg",
        chunkSizeBytes: 8 * 1024 * 1024,
        fileToken: createdFileToken,
      },
      {
        status: "uploading",
        bytesVerified: 4,
        contentType: "image/jpeg",
        chunkSizeBytes: 8 * 1024 * 1024,
        fileToken: createdFileToken,
        sessionId: "upload_session_v2_1",
      },
      {
        status: "completed",
        bytesVerified: 4,
        contentType: "image/jpeg",
        chunkSizeBytes: 8 * 1024 * 1024,
        fileToken: createdFileToken,
        sessionId: "upload_session_v2_1",
      },
    ]);
    expect(progressEvents.every((event) =>
      event.fingerprint?.startsWith("prairie.jpg:4:1780915200000:image/jpeg")
    )).toBe(true);
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/upload-sessions",
      `https://ultra.example.org/v2/upload-sessions/upload_session_v2_1/files/${createdFileToken}/chunks/0`,
      `https://ultra.example.org/v2/upload-sessions/upload_session_v2_1/files/${createdFileToken}/complete`,
    ]);
  });

  it("completes zero-byte V2 uploads without sending empty chunk requests", async () => {
    let createdFileToken = "";
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        expect(init?.method).toBe("POST");
        const body = JSON.parse(String(init?.body));
        expect(body).toMatchObject({
          total_bytes: 0,
          files: [
            {
              original_name: "empty-marker.txt",
              content_type: "text/plain",
              size_bytes: 0,
            },
          ],
        });
        createdFileToken = body.files[0].file_token;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_empty",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 0,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_empty",
                file_token: createdFileToken,
                original_name: "empty-marker.txt",
                content_type: "text/plain",
                size_bytes: 0,
                status: "pending",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:00Z",
              },
            ],
            chunks: [],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.includes("/chunks/")) {
        throw new Error(`Unexpected zero-byte chunk upload: ${url}`);
      }
      if (
        createdFileToken &&
        url ===
          `https://ultra.example.org/v2/upload-sessions/upload_session_empty/files/${createdFileToken}/complete`
      ) {
        expect(init?.method).toBe("POST");
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_empty",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "completed",
              total_bytes: 0,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              completed_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_empty",
              file_token: createdFileToken,
              resource_id: "file_empty_marker",
              original_name: "empty-marker.txt",
              content_type: "text/plain",
              size_bytes: 0,
              computed_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              completed_at: "2026-06-08T00:00:01Z",
            },
            resource: {
              file_id: "file_empty_marker",
              original_name: "empty-marker.txt",
              content_type: "text/plain",
              size_bytes: 0,
              sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              created_at: "2026-06-08T00:00:01Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const progressEvents: Array<{ status: string; bytesVerified: number; bytesCommitted: number }> = [];
    const response = await client.uploadFiles(
      [new File([], "empty-marker.txt", { type: "text/plain", lastModified: 1_780_915_200_000 })],
      {
        onProgress: (event) => {
          progressEvents.push({
            status: event.status,
            bytesVerified: event.bytesVerified,
            bytesCommitted: event.bytesCommitted,
          });
        },
      }
    );

    expect(response.uploaded[0]).toMatchObject({
      file_id: "file_empty_marker",
      size_bytes: 0,
    });
    expect(progressEvents).toEqual([
      { status: "creating", bytesVerified: 0, bytesCommitted: 0 },
      { status: "completed", bytesVerified: 0, bytesCommitted: 0 },
    ]);
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/upload-sessions",
      `https://ultra.example.org/v2/upload-sessions/upload_session_empty/files/${createdFileToken}/complete`,
    ]);
  });

  it("preserves browser folder relative paths in V2 upload sessions", async () => {
    const file = new File(["tile"], "cells.ome.tiff", {
      type: "image/tiff",
      lastModified: 1_780_915_200_000,
    });
    Object.defineProperty(file, "webkitRelativePath", {
      value: "experiment-a/day-1/cells.ome.tiff",
      configurable: true,
    });

    let createdFileToken = "";
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body));
        createdFileToken = body.files[0].file_token;
        expect(body.idempotency_key).toMatch(
          /^experiment-a\/day-1\/cells\.ome\.tiff:4:1780915200000:image\/tiff/
        );
        expect(body.browser_fingerprint).toBe(body.idempotency_key);
        expect(body.files[0]).toMatchObject({
          file_token: createdFileToken,
          original_name: "cells.ome.tiff",
          relative_path: "experiment-a/day-1/cells.ome.tiff",
          content_type: "image/tiff",
          size_bytes: 4,
        });
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 4,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_folder_1",
                file_token: createdFileToken,
                original_name: "cells.ome.tiff",
                relative_path: "experiment-a/day-1/cells.ome.tiff",
                content_type: "image/tiff",
                size_bytes: 4,
                status: "pending",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:00Z",
              },
            ],
            chunks: [],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/chunks/0`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 4,
              bytes_received: 4,
              bytes_verified: 4,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_folder_1",
              file_token: createdFileToken,
              original_name: "cells.ome.tiff",
              relative_path: "experiment-a/day-1/cells.ome.tiff",
              content_type: "image/tiff",
              size_bytes: 4,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
            },
            chunk: {
              session_id: "upload_session_folder_1",
              file_token: createdFileToken,
              chunk_index: 0,
              offset: 0,
              size_bytes: 4,
              sha256: new Headers(init?.headers).get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/complete`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "completed",
              total_bytes: 4,
              bytes_received: 4,
              bytes_verified: 4,
              bytes_committed: 4,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_folder_1",
              file_token: createdFileToken,
              resource_id: "file_folder_tile",
              original_name: "cells.ome.tiff",
              relative_path: "experiment-a/day-1/cells.ome.tiff",
              content_type: "image/tiff",
              size_bytes: 4,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
            },
            resource: {
              file_id: "file_folder_tile",
              original_name: "cells.ome.tiff",
              content_type: "image/tiff",
              size_bytes: 4,
              sha256: "folder-sha",
              created_at: "2026-06-08T00:00:02Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const progressFingerprints: string[] = [];
    const response = await client.uploadFiles([file], {
      onProgress: (event) => {
        if (event.fingerprint) {
          progressFingerprints.push(event.fingerprint);
        }
      },
    });

    expect(response.uploaded[0].file_id).toBe("file_folder_tile");
    expect(progressFingerprints.every((value) =>
      value.startsWith("experiment-a/day-1/cells.ome.tiff:4:1780915200000:image/tiff")
    )).toBe(true);
  });

  it("creates one V2 upload session manifest for multi-file folder selections", async () => {
    const firstFile = new File(["alpha"], "alpha.ome.tiff", {
      type: "image/tiff",
      lastModified: 1_780_915_200_000,
    });
    Object.defineProperty(firstFile, "webkitRelativePath", {
      value: "experiment-a/alpha.ome.tiff",
      configurable: true,
    });
    const secondFile = new File(["beta"], "beta.ome.tiff", {
      type: "image/tiff",
      lastModified: 1_780_915_201_000,
    });
    Object.defineProperty(secondFile, "webkitRelativePath", {
      value: "experiment-a/nested/beta.ome.tiff",
      configurable: true,
    });
    const fileTokens: string[] = [];
    const completedTokens: string[] = [];
    const uploadedChunkTokens: string[] = [];
    const sessionPosts: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body)) as {
          idempotency_key: string;
          total_bytes: number;
          files: Array<{
            file_token: string;
            original_name: string;
            relative_path?: string;
            size_bytes: number;
          }>;
        };
        sessionPosts.push(body);
        expect(body.total_bytes).toBe(9);
        expect(body.files).toHaveLength(2);
        expect(body.files.map((file) => file.relative_path)).toEqual([
          "experiment-a/alpha.ome.tiff",
          "experiment-a/nested/beta.ome.tiff",
        ]);
        fileTokens.splice(0, fileTokens.length, ...body.files.map((file) => file.file_token));
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_manifest",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 9,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: body.files.map((file) => ({
              session_id: "upload_session_folder_manifest",
              file_token: file.file_token,
              original_name: file.original_name,
              relative_path: file.relative_path,
              content_type: "image/tiff",
              size_bytes: file.size_bytes,
              status: "pending",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
            })),
            chunks: [],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      const chunkMatch = url.match(/\/files\/([^/]+)\/chunks\/0$/);
      if (chunkMatch) {
        const token = decodeURIComponent(chunkMatch[1]);
        uploadedChunkTokens.push(token);
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_manifest",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 9,
              bytes_received: uploadedChunkTokens.length === 1 ? 5 : 9,
              bytes_verified: uploadedChunkTokens.length === 1 ? 5 : 9,
              bytes_committed: completedTokens.length === 0 ? 0 : 5,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_folder_manifest",
              file_token: token,
              original_name: token === fileTokens[0] ? "alpha.ome.tiff" : "beta.ome.tiff",
              relative_path:
                token === fileTokens[0]
                  ? "experiment-a/alpha.ome.tiff"
                  : "experiment-a/nested/beta.ome.tiff",
              content_type: "image/tiff",
              size_bytes: token === fileTokens[0] ? 5 : 4,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
            },
            chunk: {
              session_id: "upload_session_folder_manifest",
              file_token: token,
              chunk_index: 0,
              offset: 0,
              size_bytes: token === fileTokens[0] ? 5 : 4,
              sha256: new Headers(init?.headers).get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      const completeMatch = url.match(/\/files\/([^/]+)\/complete$/);
      if (completeMatch) {
        const token = decodeURIComponent(completeMatch[1]);
        completedTokens.push(token);
        const isFirst = token === fileTokens[0];
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_manifest",
              owner_user_id: "local-user",
              source_type: "upload",
              status: completedTokens.length === 2 ? "completed" : "active",
              total_bytes: 9,
              bytes_received: 9,
              bytes_verified: 9,
              bytes_committed: completedTokens.length === 1 ? 5 : 9,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_folder_manifest",
              file_token: token,
              resource_id: isFirst ? "file_alpha" : "file_beta",
              original_name: isFirst ? "alpha.ome.tiff" : "beta.ome.tiff",
              relative_path: isFirst
                ? "experiment-a/alpha.ome.tiff"
                : "experiment-a/nested/beta.ome.tiff",
              content_type: "image/tiff",
              size_bytes: isFirst ? 5 : 4,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
            },
            resource: {
              file_id: isFirst ? "file_alpha" : "file_beta",
              original_name: isFirst ? "alpha.ome.tiff" : "beta.ome.tiff",
              content_type: "image/tiff",
              size_bytes: isFirst ? 5 : 4,
              sha256: isFirst ? "alpha-sha" : "beta-sha",
              created_at: "2026-06-08T00:00:02Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.uploadFiles([firstFile, secondFile]);

    expect(response.uploaded.map((file) => file.file_id)).toEqual(["file_alpha", "file_beta"]);
    expect(sessionPosts).toHaveLength(1);
    expect(uploadedChunkTokens).toHaveLength(fileTokens.length);
    expect(new Set(uploadedChunkTokens)).toEqual(new Set(fileTokens));
    expect(completedTokens).toHaveLength(fileTokens.length);
    expect(new Set(completedTokens)).toEqual(new Set(fileTokens));
  });

  it("does not re-upload completed files when reselecting a partially completed folder", async () => {
    const chunkSize = 8 * 1024 * 1024;
    const verifiedBetaChunk = new Uint8Array(chunkSize);
    verifiedBetaChunk.fill(3);
    const betaTail = new Uint8Array([4, 5, 6]);
    const firstFile = new File(["alpha"], "alpha.ome.tiff", {
      type: "image/tiff",
      lastModified: 1_780_915_200_000,
    });
    Object.defineProperty(firstFile, "webkitRelativePath", {
      value: "experiment-a/alpha.ome.tiff",
      configurable: true,
    });
    const secondFile = new File([verifiedBetaChunk, betaTail], "beta.ome.tiff", {
      type: "image/tiff",
      lastModified: 1_780_915_201_000,
    });
    Object.defineProperty(secondFile, "webkitRelativePath", {
      value: "experiment-a/beta.ome.tiff",
      configurable: true,
    });

    let alphaToken = "";
    let betaToken = "";
    const urls: string[] = [];
    const completedTokens: string[] = [];
    const progressEvents: Array<{ fileName: string; status: string; bytesVerified: number }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      urls.push(url);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body)) as {
          total_bytes: number;
          files: Array<{ file_token: string; original_name: string; relative_path?: string; size_bytes: number }>;
        };
        expect(body.total_bytes).toBe(firstFile.size + secondFile.size);
        alphaToken = body.files[0].file_token;
        betaToken = body.files[1].file_token;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_reselect",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: firstFile.size + secondFile.size,
              bytes_received: firstFile.size + chunkSize,
              bytes_verified: firstFile.size + chunkSize,
              bytes_committed: firstFile.size,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_folder_reselect",
                file_token: alphaToken,
                resource_id: "file_alpha_done",
                original_name: "alpha.ome.tiff",
                relative_path: "experiment-a/alpha.ome.tiff",
                content_type: "image/tiff",
                size_bytes: firstFile.size,
                status: "completed",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:01Z",
                completed_at: "2026-06-08T00:00:01Z",
              },
              {
                session_id: "upload_session_folder_reselect",
                file_token: betaToken,
                original_name: "beta.ome.tiff",
                relative_path: "experiment-a/beta.ome.tiff",
                content_type: "image/tiff",
                size_bytes: secondFile.size,
                status: "uploading",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:01Z",
              },
            ],
            chunks: [
              {
                session_id: "upload_session_folder_reselect",
                file_token: betaToken,
                chunk_index: 0,
                offset: 0,
                size_bytes: chunkSize,
                sha256: "already-verified-beta",
                status: "verified",
              },
            ],
            limits: {
              max_parallel_files: 1,
              max_parallel_chunks: 1,
              max_files_per_session: 1000,
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${alphaToken}/chunks/0`)) {
        throw new Error("completed alpha should not be re-uploaded");
      }
      if (url.endsWith(`/files/${betaToken}/chunks/0`)) {
        throw new Error("verified beta chunk should not be re-uploaded");
      }
      if (url.endsWith(`/files/${betaToken}/chunks/1`)) {
        expect(init?.method).toBe("PUT");
        expect(new Headers(init?.headers).get("x-upload-offset")).toBe(String(chunkSize));
        expect((init?.body as Blob).size).toBe(betaTail.length);
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_reselect",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: firstFile.size + secondFile.size,
              bytes_received: firstFile.size + secondFile.size,
              bytes_verified: firstFile.size + secondFile.size,
              bytes_committed: firstFile.size,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_folder_reselect",
              file_token: betaToken,
              original_name: "beta.ome.tiff",
              relative_path: "experiment-a/beta.ome.tiff",
              content_type: "image/tiff",
              size_bytes: secondFile.size,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
            },
            chunk: {
              session_id: "upload_session_folder_reselect",
              file_token: betaToken,
              chunk_index: 1,
              offset: chunkSize,
              size_bytes: betaTail.length,
              sha256: new Headers(init?.headers).get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      const completeMatch = url.match(/\/files\/([^/]+)\/complete$/);
      if (completeMatch) {
        const token = decodeURIComponent(completeMatch[1]);
        completedTokens.push(token);
        const isAlpha = token === alphaToken;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_reselect",
              owner_user_id: "local-user",
              source_type: "upload",
              status: completedTokens.includes(betaToken) ? "completed" : "active",
              total_bytes: firstFile.size + secondFile.size,
              bytes_received: firstFile.size + secondFile.size,
              bytes_verified: firstFile.size + secondFile.size,
              bytes_committed: isAlpha ? firstFile.size : firstFile.size + secondFile.size,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_folder_reselect",
              file_token: token,
              resource_id: isAlpha ? "file_alpha_done" : "file_beta_done",
              original_name: isAlpha ? "alpha.ome.tiff" : "beta.ome.tiff",
              relative_path: isAlpha ? "experiment-a/alpha.ome.tiff" : "experiment-a/beta.ome.tiff",
              content_type: "image/tiff",
              size_bytes: isAlpha ? firstFile.size : secondFile.size,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
            },
            resource: {
              file_id: isAlpha ? "file_alpha_done" : "file_beta_done",
              original_name: isAlpha ? "alpha.ome.tiff" : "beta.ome.tiff",
              content_type: "image/tiff",
              size_bytes: isAlpha ? firstFile.size : secondFile.size,
              sha256: isAlpha ? "alpha-sha" : "beta-sha",
              created_at: "2026-06-08T00:00:03Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.uploadFiles([firstFile, secondFile], {
      onProgress: (event) => {
        progressEvents.push({
          fileName: event.fileName,
          status: event.status,
          bytesVerified: event.bytesVerified,
        });
      },
    });

    expect(response.uploaded.map((file) => file.file_id)).toEqual(["file_alpha_done", "file_beta_done"]);
    expect(urls.some((url) => url.endsWith(`/files/${alphaToken}/chunks/0`))).toBe(false);
    expect(urls.some((url) => url.endsWith(`/files/${betaToken}/chunks/0`))).toBe(false);
    expect(urls).toContain(
      `https://ultra.example.org/v2/upload-sessions/upload_session_folder_reselect/files/${betaToken}/chunks/1`
    );
    expect(new Set(completedTokens)).toEqual(new Set([alphaToken, betaToken]));
    expect(progressEvents).toContainEqual({
      fileName: "alpha.ome.tiff",
      status: "completed",
      bytesVerified: firstFile.size,
    });
    expect(progressEvents).toContainEqual({
      fileName: "beta.ome.tiff",
      status: "completed",
      bytesVerified: secondFile.size,
    });
  });

  it("does not mark completed files failed when another file in a folder upload fails", async () => {
    const firstFile = new File(["alpha"], "alpha.ome.tiff", {
      type: "image/tiff",
      lastModified: 1_780_915_200_000,
    });
    Object.defineProperty(firstFile, "webkitRelativePath", {
      value: "experiment-a/alpha.ome.tiff",
      configurable: true,
    });
    const secondFile = new File(["beta"], "beta.ome.tiff", {
      type: "image/tiff",
      lastModified: 1_780_915_201_000,
    });
    Object.defineProperty(secondFile, "webkitRelativePath", {
      value: "experiment-a/beta.ome.tiff",
      configurable: true,
    });
    const fileTokens: string[] = [];
    const progressEvents: Array<{ fileName: string; status: string; bytesVerified: number }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body)) as {
          files: Array<{ file_token: string; original_name: string; relative_path?: string; size_bytes: number }>;
          total_bytes: number;
        };
        expect(body.total_bytes).toBe(9);
        fileTokens.splice(0, fileTokens.length, ...body.files.map((file) => file.file_token));
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_partial_failure",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 9,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: body.files.map((file) => ({
              session_id: "upload_session_folder_partial_failure",
              file_token: file.file_token,
              original_name: file.original_name,
              relative_path: file.relative_path,
              content_type: "image/tiff",
              size_bytes: file.size_bytes,
              status: "pending",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
            })),
            chunks: [],
            limits: {
              max_parallel_files: 1,
              max_parallel_chunks: 1,
              max_files_per_session: 1000,
            },
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      const chunkMatch = url.match(/\/files\/([^/]+)\/chunks\/0$/);
      if (chunkMatch) {
        const token = decodeURIComponent(chunkMatch[1]);
        if (token === fileTokens[1]) {
          return new Response(JSON.stringify({ error: "connection dropped" }), {
            status: 400,
            headers: { "Content-Type": "application/json" },
          });
        }
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_partial_failure",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 9,
              bytes_received: 5,
              bytes_verified: 5,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_folder_partial_failure",
              file_token: token,
              original_name: "alpha.ome.tiff",
              relative_path: "experiment-a/alpha.ome.tiff",
              content_type: "image/tiff",
              size_bytes: 5,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
            },
            chunk: {
              session_id: "upload_session_folder_partial_failure",
              file_token: token,
              chunk_index: 0,
              offset: 0,
              size_bytes: 5,
              sha256: new Headers(init?.headers).get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${fileTokens[0]}/complete`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_folder_partial_failure",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 9,
              bytes_received: 5,
              bytes_verified: 5,
              bytes_committed: 5,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_folder_partial_failure",
              file_token: fileTokens[0],
              resource_id: "file_alpha",
              original_name: "alpha.ome.tiff",
              relative_path: "experiment-a/alpha.ome.tiff",
              content_type: "image/tiff",
              size_bytes: 5,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
            },
            resource: {
              file_id: "file_alpha",
              original_name: "alpha.ome.tiff",
              content_type: "image/tiff",
              size_bytes: 5,
              sha256: "alpha-sha",
              created_at: "2026-06-08T00:00:02Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    await expect(
      client.uploadFiles([firstFile, secondFile], {
        onProgress: (event) => {
          progressEvents.push({
            fileName: event.fileName,
            status: event.status,
            bytesVerified: event.bytesVerified,
          });
        },
      })
    ).rejects.toThrow("Request failed with status 400");

    expect(progressEvents.filter((event) => event.fileName === "alpha.ome.tiff")).toEqual([
      { fileName: "alpha.ome.tiff", status: "creating", bytesVerified: 0 },
      { fileName: "alpha.ome.tiff", status: "uploading", bytesVerified: 5 },
      { fileName: "alpha.ome.tiff", status: "completed", bytesVerified: 5 },
    ]);
    expect(progressEvents).toContainEqual({
      fileName: "beta.ome.tiff",
      status: "failed",
      bytesVerified: 0,
    });
  });

  it("uploads files inside a batch V2 session with bounded file parallelism", async () => {
    const files = Array.from({ length: 4 }, (_value, index) => {
      const file = new File([`f${index}`], `tile-${index}.ome.tiff`, {
        type: "image/tiff",
        lastModified: 1_780_915_200_000 + index,
      });
      Object.defineProperty(file, "webkitRelativePath", {
        value: `experiment-a/tile-${index}.ome.tiff`,
        configurable: true,
      });
      return file;
    });
    const fileTokens: string[] = [];
    const completedTokens: string[] = [];
    const uploadedChunkTokens: string[] = [];
    let activeChunkUploads = 0;
    let maxActiveChunkUploads = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body)) as {
          files: Array<{ file_token: string; original_name: string; relative_path?: string; size_bytes: number }>;
          total_bytes: number;
        };
        expect(body.total_bytes).toBe(8);
        expect(body.files).toHaveLength(4);
        fileTokens.splice(0, fileTokens.length, ...body.files.map((file) => file.file_token));
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_parallel_files",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 8,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: body.files.map((file) => ({
              session_id: "upload_session_parallel_files",
              file_token: file.file_token,
              original_name: file.original_name,
              relative_path: file.relative_path,
              content_type: "image/tiff",
              size_bytes: file.size_bytes,
              status: "pending",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
            })),
            chunks: [],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      const chunkMatch = url.match(/\/files\/([^/]+)\/chunks\/0$/);
      if (chunkMatch) {
        const token = decodeURIComponent(chunkMatch[1]);
        activeChunkUploads += 1;
        maxActiveChunkUploads = Math.max(maxActiveChunkUploads, activeChunkUploads);
        uploadedChunkTokens.push(token);
        await new Promise((resolve) => setTimeout(resolve, 25));
        activeChunkUploads -= 1;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_parallel_files",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: 8,
              bytes_received: Math.min(8, uploadedChunkTokens.length * 2),
              bytes_verified: Math.min(8, uploadedChunkTokens.length * 2),
              bytes_committed: Math.min(8, completedTokens.length * 2),
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_parallel_files",
              file_token: token,
              original_name: "tile.ome.tiff",
              content_type: "image/tiff",
              size_bytes: 2,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
            },
            chunk: {
              session_id: "upload_session_parallel_files",
              file_token: token,
              chunk_index: 0,
              offset: 0,
              size_bytes: 2,
              sha256: new Headers(init?.headers).get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      const completeMatch = url.match(/\/files\/([^/]+)\/complete$/);
      if (completeMatch) {
        const token = decodeURIComponent(completeMatch[1]);
        completedTokens.push(token);
        const fileIndex = fileTokens.indexOf(token);
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_parallel_files",
              owner_user_id: "local-user",
              source_type: "upload",
              status: completedTokens.length === 4 ? "completed" : "active",
              total_bytes: 8,
              bytes_received: 8,
              bytes_verified: 8,
              bytes_committed: Math.min(8, completedTokens.length * 2),
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_parallel_files",
              file_token: token,
              resource_id: `file_tile_${fileIndex}`,
              original_name: `tile-${fileIndex}.ome.tiff`,
              content_type: "image/tiff",
              size_bytes: 2,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
            },
            resource: {
              file_id: `file_tile_${fileIndex}`,
              original_name: `tile-${fileIndex}.ome.tiff`,
              content_type: "image/tiff",
              size_bytes: 2,
              sha256: `tile-${fileIndex}-sha`,
              created_at: "2026-06-08T00:00:02Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.uploadFiles(files);

    expect(response.uploaded.map((file) => file.file_id)).toEqual([
      "file_tile_0",
      "file_tile_1",
      "file_tile_2",
      "file_tile_3",
    ]);
    expect(maxActiveChunkUploads).toBeGreaterThan(1);
    expect(maxActiveChunkUploads).toBeLessThanOrEqual(4);
    expect(new Set(uploadedChunkTokens)).toEqual(new Set(fileTokens));
    expect(new Set(completedTokens)).toEqual(new Set(fileTokens));
  });

  it("skips server-verified chunks when resuming V2 upload sessions", async () => {
    const chunkSize = 8 * 1024 * 1024;
    const firstChunk = new Uint8Array(chunkSize);
    firstChunk.fill(7);
    const tailChunk = new Uint8Array([1, 2, 3]);
    const file = new File([firstChunk, tailChunk], "large-brain.nii", {
      type: "application/x-nifti",
      lastModified: 1_780_915_200_000,
    });
    let createdFileToken = "";
    const urls: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      urls.push(url);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body));
        createdFileToken = body.files[0].file_token;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_resume_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: chunkSize,
              bytes_verified: chunkSize,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_resume_1",
                file_token: createdFileToken,
                original_name: "large-brain.nii",
                content_type: "application/x-nifti",
                size_bytes: file.size,
                status: "uploading",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:01Z",
              },
            ],
            chunks: [
              {
                session_id: "upload_session_resume_1",
                file_token: createdFileToken,
                chunk_index: 0,
                offset: 0,
                size_bytes: chunkSize,
                sha256: "already-verified",
                status: "verified",
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/chunks/0`)) {
        throw new Error("chunk 0 should not be re-uploaded");
      }
      if (url.endsWith(`/files/${createdFileToken}/chunks/1`)) {
        const headers = new Headers(init?.headers);
        expect(headers.get("x-upload-offset")).toBe(String(chunkSize));
        expect((init?.body as Blob).size).toBe(tailChunk.length);
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_resume_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: file.size,
              bytes_verified: file.size,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_resume_1",
              file_token: createdFileToken,
              original_name: "large-brain.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
            },
            chunk: {
              session_id: "upload_session_resume_1",
              file_token: createdFileToken,
              chunk_index: 1,
              offset: chunkSize,
              size_bytes: tailChunk.length,
              sha256: headers.get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/complete`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_resume_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "completed",
              total_bytes: file.size,
              bytes_received: file.size,
              bytes_verified: file.size,
              bytes_committed: file.size,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_resume_1",
              file_token: createdFileToken,
              resource_id: "file_large_brain",
              original_name: "large-brain.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
            },
            resource: {
              file_id: "file_large_brain",
              original_name: "large-brain.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              sha256: "final-sha",
              created_at: "2026-06-08T00:00:03Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.uploadFiles([file]);

    expect(response.uploaded[0].file_id).toBe("file_large_brain");
    expect(urls.some((url) => url.endsWith(`/files/${createdFileToken}/chunks/0`))).toBe(false);
    expect(urls).toContain(
      `https://ultra.example.org/v2/upload-sessions/upload_session_resume_1/files/${createdFileToken}/chunks/1`
    );
  });

  it("uses an explicit V2 upload session when resuming a reselected file", async () => {
    const chunkSize = 8 * 1024 * 1024;
    const firstChunk = new Uint8Array(chunkSize);
    firstChunk.fill(11);
    const tailChunk = new Uint8Array([7, 8, 9]);
    const file = new File([firstChunk, tailChunk], "resume-brain.nii", {
      type: "application/x-nifti",
      lastModified: 1_780_915_200_000,
    });
    const resumeFileToken = "file-original-folder-token";
    const urls: string[] = [];
    const progressEvents: Array<{ id: string; status: string; fileToken?: string; sessionId?: string; bytesVerified: number }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      urls.push(url);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        throw new Error("resume should not create a fresh upload session");
      }
      if (url === "https://ultra.example.org/v2/upload-sessions/upload_session_reselect_1") {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_reselect_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: chunkSize,
              bytes_verified: chunkSize,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_reselect_1",
                file_token: resumeFileToken,
                original_name: "resume-brain.nii",
                content_type: "application/x-nifti",
                size_bytes: file.size,
                status: "uploading",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:01Z",
              },
            ],
            chunks: [
              {
                session_id: "upload_session_reselect_1",
                file_token: resumeFileToken,
                chunk_index: 0,
                offset: 0,
                size_bytes: chunkSize,
                sha256: "already-verified",
                status: "verified",
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${resumeFileToken}/chunks/0`)) {
        throw new Error("verified chunk 0 should not be re-uploaded");
      }
      if (url.endsWith(`/files/${resumeFileToken}/chunks/1`)) {
        expect(init?.method).toBe("PUT");
        expect(new Headers(init?.headers).get("x-upload-offset")).toBe(String(chunkSize));
        expect((init?.body as Blob).size).toBe(tailChunk.length);
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_reselect_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: file.size,
              bytes_verified: file.size,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_reselect_1",
              file_token: resumeFileToken,
              original_name: "resume-brain.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
            },
            chunk: {
              session_id: "upload_session_reselect_1",
              file_token: resumeFileToken,
              chunk_index: 1,
              offset: chunkSize,
              size_bytes: tailChunk.length,
              sha256: new Headers(init?.headers).get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${resumeFileToken}/complete`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_reselect_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "completed",
              total_bytes: file.size,
              bytes_received: file.size,
              bytes_verified: file.size,
              bytes_committed: file.size,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_reselect_1",
              file_token: resumeFileToken,
              resource_id: "file_resume_brain",
              original_name: "resume-brain.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
            },
            resource: {
              file_id: "file_resume_brain",
              original_name: "resume-brain.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              sha256: "resume-sha",
              created_at: "2026-06-08T00:00:03Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.uploadFiles([file], {
      resumeSession: {
        sessionId: "upload_session_reselect_1",
        fileToken: resumeFileToken,
        progressId: "upload-progress-reselect",
      },
      onProgress: (event) => {
        progressEvents.push({
          id: event.id,
          status: event.status,
          fileToken: event.fileToken,
          sessionId: event.sessionId,
          bytesVerified: event.bytesVerified,
        });
      },
    });

    expect(response.uploaded[0].file_id).toBe("file_resume_brain");
    expect(urls[0]).toBe("https://ultra.example.org/v2/upload-sessions/upload_session_reselect_1");
    expect(urls.some((url) => url === "https://ultra.example.org/v2/upload-sessions")).toBe(false);
    expect(urls.some((url) => url.endsWith(`/files/${resumeFileToken}/chunks/0`))).toBe(false);
    expect(progressEvents).toMatchObject([
      {
        id: "upload-progress-reselect",
        status: "creating",
        fileToken: resumeFileToken,
        sessionId: "upload_session_reselect_1",
        bytesVerified: 0,
      },
      {
        id: "upload-progress-reselect",
        status: "uploading",
        fileToken: resumeFileToken,
        sessionId: "upload_session_reselect_1",
        bytesVerified: file.size,
      },
      {
        id: "upload-progress-reselect",
        status: "completed",
        fileToken: resumeFileToken,
        sessionId: "upload_session_reselect_1",
        bytesVerified: file.size,
      },
    ]);
  });

  it("preserves verified bytes when a later V2 chunk upload fails", async () => {
    const chunkSize = 8 * 1024 * 1024;
    const firstChunk = new Uint8Array(chunkSize);
    firstChunk.fill(9);
    const tailChunk = new Uint8Array([4, 5, 6]);
    const file = new File([firstChunk, tailChunk], "interrupted-brain.nii", {
      type: "application/x-nifti",
      lastModified: 1_780_915_200_000,
    });
    let createdFileToken = "";
    const progressEvents: Array<{ status: string; bytesVerified: number; sessionId?: string }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body));
        createdFileToken = body.files[0].file_token;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_interrupted_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_interrupted_1",
                file_token: createdFileToken,
                original_name: "interrupted-brain.nii",
                content_type: "application/x-nifti",
                size_bytes: file.size,
                status: "pending",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:00Z",
              },
            ],
            chunks: [],
            limits: {
              max_parallel_files: 1,
              max_parallel_chunks: 1,
              max_files_per_session: 1000,
            },
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/chunks/0`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_interrupted_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: chunkSize,
              bytes_verified: chunkSize,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_interrupted_1",
              file_token: createdFileToken,
              original_name: "interrupted-brain.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
            },
            chunk: {
              session_id: "upload_session_interrupted_1",
              file_token: createdFileToken,
              chunk_index: 0,
              offset: 0,
              size_bytes: chunkSize,
              sha256: new Headers(init?.headers).get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/chunks/1`)) {
        return new Response(JSON.stringify({ error: "connection dropped" }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.endsWith(`/files/${createdFileToken}/complete`)) {
        throw new Error("interrupted upload should not complete");
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    await expect(
      client.uploadFiles([file], {
        onProgress: (event) => {
          progressEvents.push({
            status: event.status,
            bytesVerified: event.bytesVerified,
            sessionId: event.sessionId,
          });
        },
      })
    ).rejects.toThrow("Request failed with status 400");

    expect(progressEvents).toMatchObject([
      { status: "creating", bytesVerified: 0 },
      { status: "uploading", bytesVerified: chunkSize, sessionId: "upload_session_interrupted_1" },
      { status: "failed", bytesVerified: chunkSize, sessionId: "upload_session_interrupted_1" },
    ]);
  });

  it("reopens paused V2 upload sessions before sending resumed chunks", async () => {
    const file = new File(["paused bytes"], "paused-brain.nii", {
      type: "application/x-nifti",
      lastModified: 1_780_915_200_000,
    });
    let createdFileToken = "";
    const urls: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      urls.push(url);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body));
        createdFileToken = body.files[0].file_token;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_paused_resume",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "paused",
              total_bytes: file.size,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_paused_resume",
                file_token: createdFileToken,
                original_name: "paused-brain.nii",
                content_type: "application/x-nifti",
                size_bytes: file.size,
                status: "uploading",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:01Z",
              },
            ],
            chunks: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/upload-sessions/upload_session_paused_resume/resume") {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_paused_resume",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:02Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_paused_resume",
                file_token: createdFileToken,
                original_name: "paused-brain.nii",
                content_type: "application/x-nifti",
                size_bytes: file.size,
                status: "uploading",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:02Z",
              },
            ],
            chunks: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/chunks/0`)) {
        const resumeIndex = urls.indexOf("https://ultra.example.org/v2/upload-sessions/upload_session_paused_resume/resume");
        expect(resumeIndex).toBeGreaterThan(-1);
        expect(urls.length - 1).toBeGreaterThan(resumeIndex);
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_paused_resume",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: file.size,
              bytes_verified: file.size,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_paused_resume",
              file_token: createdFileToken,
              original_name: "paused-brain.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
            },
            chunk: {
              session_id: "upload_session_paused_resume",
              file_token: createdFileToken,
              chunk_index: 0,
              offset: 0,
              size_bytes: file.size,
              sha256: new Headers(init?.headers).get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/complete`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_paused_resume",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "completed",
              total_bytes: file.size,
              bytes_received: file.size,
              bytes_verified: file.size,
              bytes_committed: file.size,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:04Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_paused_resume",
              file_token: createdFileToken,
              resource_id: "file_paused_brain",
              original_name: "paused-brain.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:04Z",
            },
            resource: {
              file_id: "file_paused_brain",
              original_name: "paused-brain.nii",
              content_type: "application/x-nifti",
              size_bytes: file.size,
              sha256: "paused-final-sha",
              created_at: "2026-06-08T00:00:04Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.uploadFiles([file]);

    expect(response.uploaded[0].file_id).toBe("file_paused_brain");
    expect(urls).toContain("https://ultra.example.org/v2/upload-sessions/upload_session_paused_resume/resume");
  });

  it("retries transient V2 chunk failures without restarting the upload session", async () => {
    const chunkSize = 8 * 1024 * 1024;
    const file = new File(
      [new Uint8Array(chunkSize), new Uint8Array([4, 5, 6])],
      "starlink-field-stack.ome.tiff",
      {
        type: "image/tiff",
        lastModified: 1_780_915_200_000,
      }
    );
    let createdFileToken = "";
    let sessionCreateCount = 0;
    const chunkAttempts = new Map<number, number>();
    const progressEvents: Array<{ status: string; bytesVerified: number }> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        sessionCreateCount += 1;
        const body = JSON.parse(String(init?.body));
        createdFileToken = body.files[0].file_token;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_starlink_retry",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_starlink_retry",
                file_token: createdFileToken,
                original_name: "starlink-field-stack.ome.tiff",
                content_type: "image/tiff",
                size_bytes: file.size,
                status: "pending",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:00Z",
              },
            ],
            chunks: [],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      const chunkMatch = url.match(/\/chunks\/(\d+)$/);
      if (chunkMatch) {
        const chunkIndex = Number(chunkMatch[1]);
        const attempt = (chunkAttempts.get(chunkIndex) ?? 0) + 1;
        chunkAttempts.set(chunkIndex, attempt);
        if (chunkIndex === 1 && attempt === 1) {
          return new Response(JSON.stringify({ error: "temporary satellite link drop" }), {
            status: 503,
            headers: { "Content-Type": "application/json" },
          });
        }
        const headers = new Headers(init?.headers);
        const chunk = init?.body as Blob;
        const bytesVerified = chunkIndex === 0 ? chunkSize : file.size;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_starlink_retry",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: bytesVerified,
              bytes_verified: bytesVerified,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_starlink_retry",
              file_token: createdFileToken,
              original_name: "starlink-field-stack.ome.tiff",
              content_type: "image/tiff",
              size_bytes: file.size,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
            },
            chunk: {
              session_id: "upload_session_starlink_retry",
              file_token: createdFileToken,
              chunk_index: chunkIndex,
              offset: Number(headers.get("x-upload-offset")),
              size_bytes: chunk.size,
              sha256: headers.get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/complete`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_starlink_retry",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "completed",
              total_bytes: file.size,
              bytes_received: file.size,
              bytes_verified: file.size,
              bytes_committed: file.size,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_starlink_retry",
              file_token: createdFileToken,
              resource_id: "file_starlink_field_stack",
              original_name: "starlink-field-stack.ome.tiff",
              content_type: "image/tiff",
              size_bytes: file.size,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
            },
            resource: {
              file_id: "file_starlink_field_stack",
              original_name: "starlink-field-stack.ome.tiff",
              content_type: "image/tiff",
              size_bytes: file.size,
              sha256: "final-starlink-sha",
              created_at: "2026-06-08T00:00:03Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.uploadFiles([file], {
      onProgress: (event) => {
        progressEvents.push({ status: event.status, bytesVerified: event.bytesVerified });
      },
    });

    expect(response.uploaded[0].file_id).toBe("file_starlink_field_stack");
    expect(sessionCreateCount).toBe(1);
    expect(chunkAttempts.get(0)).toBe(1);
    expect(chunkAttempts.get(1)).toBe(2);
    expect(progressEvents[progressEvents.length - 1]).toEqual({
      status: "completed",
      bytesVerified: file.size,
    });
    expect(progressEvents.some((event) => event.status === "failed")).toBe(false);
  });

  it("uploads missing V2 chunks with bounded parallelism", async () => {
    const chunkSize = 8 * 1024 * 1024;
    const file = new File(
      [new Uint8Array(chunkSize), new Uint8Array(chunkSize), new Uint8Array([9, 8, 7])],
      "field-stack.ome.tiff",
      {
        type: "image/tiff",
        lastModified: 1_780_915_200_000,
      }
    );
    let createdFileToken = "";
    let activeChunkUploads = 0;
    let maxActiveChunkUploads = 0;
    const uploadedChunkIndexes: number[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body));
        createdFileToken = body.files[0].file_token;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_parallel_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_parallel_1",
                file_token: createdFileToken,
                original_name: "field-stack.ome.tiff",
                content_type: "image/tiff",
                size_bytes: file.size,
                status: "pending",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:00Z",
              },
            ],
            chunks: [],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      const chunkMatch = url.match(/\/chunks\/(\d+)$/);
      if (chunkMatch) {
        const chunkIndex = Number(chunkMatch[1]);
        const headers = new Headers(init?.headers);
        const chunk = init?.body as Blob;
        activeChunkUploads += 1;
        maxActiveChunkUploads = Math.max(maxActiveChunkUploads, activeChunkUploads);
        uploadedChunkIndexes.push(chunkIndex);
        await new Promise((resolve) => setTimeout(resolve, 25));
        activeChunkUploads -= 1;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_parallel_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: Math.min(file.size, (chunkIndex + 1) * chunkSize),
              bytes_verified: Math.min(file.size, (chunkIndex + 1) * chunkSize),
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_parallel_1",
              file_token: createdFileToken,
              original_name: "field-stack.ome.tiff",
              content_type: "image/tiff",
              size_bytes: file.size,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
            },
            chunk: {
              session_id: "upload_session_parallel_1",
              file_token: createdFileToken,
              chunk_index: chunkIndex,
              offset: Number(headers.get("x-upload-offset")),
              size_bytes: chunk.size,
              sha256: headers.get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/complete`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_parallel_1",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "completed",
              total_bytes: file.size,
              bytes_received: file.size,
              bytes_verified: file.size,
              bytes_committed: file.size,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_parallel_1",
              file_token: createdFileToken,
              resource_id: "file_parallel_field_stack",
              original_name: "field-stack.ome.tiff",
              content_type: "image/tiff",
              size_bytes: file.size,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
            },
            resource: {
              file_id: "file_parallel_field_stack",
              original_name: "field-stack.ome.tiff",
              content_type: "image/tiff",
              size_bytes: file.size,
              sha256: "final-parallel-sha",
              created_at: "2026-06-08T00:00:03Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.uploadFiles([file]);

    expect(response.uploaded[0].file_id).toBe("file_parallel_field_stack");
    expect(uploadedChunkIndexes.sort()).toEqual([0, 1, 2]);
    expect(maxActiveChunkUploads).toBeGreaterThan(1);
    expect(maxActiveChunkUploads).toBeLessThanOrEqual(4);
  });

  it("honors V2 upload-session chunk concurrency limits from the server", async () => {
    const chunkSize = 8 * 1024 * 1024;
    const file = new File(
      [
        new Uint8Array(chunkSize),
        new Uint8Array(chunkSize),
        new Uint8Array(chunkSize),
        new Uint8Array([7, 6, 5]),
      ],
      "server-limited-stack.ome.tiff",
      {
        type: "image/tiff",
        lastModified: 1_780_915_200_000,
      }
    );
    let createdFileToken = "";
    let activeChunkUploads = 0;
    let maxActiveChunkUploads = 0;
    const uploadedChunkIndexes: number[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/upload-sessions") {
        const body = JSON.parse(String(init?.body));
        createdFileToken = body.files[0].file_token;
        return new Response(
          JSON.stringify({
            limits: {
              max_parallel_chunks: 2,
              max_parallel_files: 4,
              max_files_per_session: 10000,
            },
            session: {
              session_id: "upload_session_server_limited",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: 0,
              bytes_verified: 0,
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            files: [
              {
                session_id: "upload_session_server_limited",
                file_token: createdFileToken,
                original_name: "server-limited-stack.ome.tiff",
                content_type: "image/tiff",
                size_bytes: file.size,
                status: "pending",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:00Z",
              },
            ],
            chunks: [],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      const chunkMatch = url.match(/\/chunks\/(\d+)$/);
      if (chunkMatch) {
        const chunkIndex = Number(chunkMatch[1]);
        const headers = new Headers(init?.headers);
        const chunk = init?.body as Blob;
        activeChunkUploads += 1;
        maxActiveChunkUploads = Math.max(maxActiveChunkUploads, activeChunkUploads);
        uploadedChunkIndexes.push(chunkIndex);
        await new Promise((resolve) => setTimeout(resolve, 25));
        activeChunkUploads -= 1;
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_server_limited",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "active",
              total_bytes: file.size,
              bytes_received: Math.min(file.size, (chunkIndex + 1) * chunkSize),
              bytes_verified: Math.min(file.size, (chunkIndex + 1) * chunkSize),
              bytes_committed: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_server_limited",
              file_token: createdFileToken,
              original_name: "server-limited-stack.ome.tiff",
              content_type: "image/tiff",
              size_bytes: file.size,
              status: "uploading",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
            },
            chunk: {
              session_id: "upload_session_server_limited",
              file_token: createdFileToken,
              chunk_index: chunkIndex,
              offset: Number(headers.get("x-upload-offset")),
              size_bytes: chunk.size,
              sha256: headers.get("x-upload-chunk-sha256"),
              status: "verified",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith(`/files/${createdFileToken}/complete`)) {
        return new Response(
          JSON.stringify({
            session: {
              session_id: "upload_session_server_limited",
              owner_user_id: "local-user",
              source_type: "upload",
              status: "completed",
              total_bytes: file.size,
              bytes_received: file.size,
              bytes_verified: file.size,
              bytes_committed: file.size,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
              metadata: {},
            },
            file: {
              session_id: "upload_session_server_limited",
              file_token: createdFileToken,
              resource_id: "file_server_limited_stack",
              original_name: "server-limited-stack.ome.tiff",
              content_type: "image/tiff",
              size_bytes: file.size,
              status: "completed",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:03Z",
            },
            resource: {
              file_id: "file_server_limited_stack",
              original_name: "server-limited-stack.ome.tiff",
              content_type: "image/tiff",
              size_bytes: file.size,
              sha256: "final-server-limited-sha",
              created_at: "2026-06-08T00:00:03Z",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.uploadFiles([file]);

    expect(response.uploaded[0].file_id).toBe("file_server_limited_stack");
    expect(uploadedChunkIndexes.sort()).toEqual([0, 1, 2, 3]);
    expect(maxActiveChunkUploads).toBeGreaterThan(1);
    expect(maxActiveChunkUploads).toBeLessThanOrEqual(2);
  });

  it("lists resources through V2 without probing legacy resource routes", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/resources?limit=50&offset=0&q=prairie&kind=image&sharing=shared_with_me&processing_status=metadata_ready&status=deleted&tags=nph%2Cunder+70&descriptors=nph%2Cventriculomegaly&metadata_filter=label%3Aeq%3ANPH&metadata_filter=subject_age%3Alt%3A70&created_after=2026-06-02&created_before=2026-06-04") {
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
    const response = await client.listResources({
      limit: 50,
      query: "prairie",
      kind: "image",
      sharing: "shared_with_me",
      processingStatus: "metadata_ready",
      status: "deleted",
      tags: [" nph ", "under 70", "nph"],
      descriptors: [" nph ", "ventriculomegaly", "nph"],
      metadataFilters: [
        { path: "label", operator: "eq", value: "NPH" },
        { path: "subject_age", operator: "lt", value: 70 },
      ],
      createdAfter: "2026-06-02",
      createdBefore: "2026-06-04",
    });

    expect(response.resources[0].file_id).toBe("file_v2_image");
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/resources?limit=50&offset=0&q=prairie&kind=image&sharing=shared_with_me&processing_status=metadata_ready&status=deleted&tags=nph%2Cunder+70&descriptors=nph%2Cventriculomegaly&metadata_filter=label%3Aeq%3ANPH&metadata_filter=subject_age%3Alt%3A70&created_after=2026-06-02&created_before=2026-06-04",
    ]);
  });

  it("bulk tags resources through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/resources/tags/bulk") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({
          resource_ids: ["file_a", "file_b"],
          tags: ["NPH", "Under 70"],
          metadata: { source: "resources_bulk_tag_panel" },
        });
        return new Response(
          JSON.stringify({
            count: 2,
            resources: [
              {
                file_id: "file_a",
                original_name: "a.nii.gz",
                size_bytes: 10,
                sha256: "sha-a",
                created_at: "2026-06-08T00:00:00Z",
                source_type: "upload",
                resource_kind: "file",
                has_thumbnail: false,
                tags: ["NPH", "Under 70"],
              },
              {
                file_id: "file_b",
                original_name: "b.nii.gz",
                size_bytes: 11,
                sha256: "sha-b",
                created_at: "2026-06-08T00:00:01Z",
                source_type: "upload",
                resource_kind: "file",
                has_thumbnail: false,
                tags: ["NPH", "Under 70"],
              },
            ],
            events: [
              {
                event_id: "resource_event_a",
                resource_id: "file_a",
                event_type: "resource.tagged",
                ts: "2026-06-08T00:00:02Z",
                metadata: { tags_added: ["NPH", "Under 70"] },
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
    const response = await client.bulkTagResources({
      resource_ids: ["file_a", "file_b"],
      tags: ["NPH", "Under 70"],
      metadata: { source: "resources_bulk_tag_panel" },
    });

    expect(response.count).toBe(2);
    expect(response.resources.map((resource) => resource.tags)).toEqual([
      ["NPH", "Under 70"],
      ["NPH", "Under 70"],
    ]);
  });

  it("patches resource metadata through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/resources/file_a") {
        expect(init?.method).toBe("PATCH");
        expect(JSON.parse(String(init?.body))).toEqual({
          metadata: {
            cohort: "NPH",
            review: { status: "checked" },
          },
        });
        return new Response(
          JSON.stringify({
            resource: {
              file_id: "file_a",
              original_name: "a.nii.gz",
              size_bytes: 10,
              sha256: "sha-a",
              created_at: "2026-06-08T00:00:00Z",
              source_type: "upload",
              resource_kind: "file",
              has_thumbnail: false,
              metadata: {
                source_label: "raw",
                cohort: "NPH",
                review: { reader: "lab-a", status: "checked" },
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
    const response = await client.patchResourceMetadata("file_a", {
      cohort: "NPH",
      review: { status: "checked" },
    });

    expect(response.resource.metadata).toMatchObject({
      source_label: "raw",
      cohort: "NPH",
      review: { reader: "lab-a", status: "checked" },
    });
  });

  it("manages resource share grants through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/resources/file_a/shares?limit=25&status=active") {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            resource_id: "file_a",
            count: 1,
            grants: [
              {
                grant_id: "resource_grant_bob",
                resource_id: "file_a",
                owner_user_id: "alice",
                grantee_user_id: "bob",
                grantee_org_id: "org-b",
                role: "read",
                status: "active",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:00Z",
                metadata: {},
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/resources/file_a/shares") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({
          grantee_user_id: "bob",
          grantee_org_id: "org-b",
          role: "read",
          metadata: { reason: "collaborative review" },
        });
        return new Response(
          JSON.stringify({
            grant: {
              grant_id: "resource_grant_bob",
              resource_id: "file_a",
              owner_user_id: "alice",
              grantee_user_id: "bob",
              grantee_org_id: "org-b",
              role: "read",
              status: "active",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/resources/file_a/shares/resource_grant_bob") {
        expect(init?.method).toBe("DELETE");
        return new Response(
          JSON.stringify({
            grant: {
              grant_id: "resource_grant_bob",
              resource_id: "file_a",
              owner_user_id: "alice",
              grantee_user_id: "bob",
              grantee_org_id: "org-b",
              role: "read",
              status: "revoked",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:01:00Z",
              revoked_at: "2026-06-08T00:01:00Z",
              metadata: {},
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/resources/shares/bulk") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({
          resource_ids: ["file_a", "file_b"],
          grantee_user_id: "charlie",
          grantee_org_id: "org-c",
          role: "read",
          metadata: { reason: "bulk review" },
        });
        return new Response(
          JSON.stringify({
            count: 2,
            grants: [
              {
                grant_id: "resource_grant_charlie_a",
                resource_id: "file_a",
                owner_user_id: "alice",
                grantee_user_id: "charlie",
                grantee_org_id: "org-c",
                role: "read",
                status: "active",
                created_at: "2026-06-08T00:02:00Z",
                updated_at: "2026-06-08T00:02:00Z",
                metadata: {},
              },
              {
                grant_id: "resource_grant_charlie_b",
                resource_id: "file_b",
                owner_user_id: "alice",
                grantee_user_id: "charlie",
                grantee_org_id: "org-c",
                role: "read",
                status: "active",
                created_at: "2026-06-08T00:02:00Z",
                updated_at: "2026-06-08T00:02:00Z",
                metadata: {},
              },
            ],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/resource-collections/collection_nph/shares") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({
          grantee_user_id: "dana",
          grantee_org_id: "org-d",
          role: "read",
          metadata: { reason: "folder review" },
        });
        return new Response(
          JSON.stringify({
            count: 2,
            collection: {
              collection_id: "collection_nph",
              owner_user_id: "alice",
              owner_org_id: "org-a",
              name: "NPH folder",
              collection_type: "folder",
              status: "active",
              resource_count: 2,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:02:00Z",
              metadata: {},
            },
            grants: [
              {
                grant_id: "resource_grant_dana_a",
                resource_id: "file_a",
                owner_user_id: "alice",
                grantee_user_id: "dana",
                grantee_org_id: "org-d",
                role: "read",
                status: "active",
                created_at: "2026-06-08T00:03:00Z",
                updated_at: "2026-06-08T00:03:00Z",
                metadata: {},
              },
              {
                grant_id: "resource_grant_dana_b",
                resource_id: "file_b",
                owner_user_id: "alice",
                grantee_user_id: "dana",
                grantee_org_id: "org-d",
                role: "read",
                status: "active",
                created_at: "2026-06-08T00:03:00Z",
                updated_at: "2026-06-08T00:03:00Z",
                metadata: {},
              },
            ],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const listed = await client.listResourceShareGrants("file_a", { limit: 25, status: "active" });
    const created = await client.createResourceShareGrant("file_a", {
      grantee_user_id: "bob",
      grantee_org_id: "org-b",
      role: "read",
      metadata: { reason: "collaborative review" },
    });
    const revoked = await client.revokeResourceShareGrant("file_a", "resource_grant_bob");
    const bulkCreated = await client.createResourceShareGrants({
      resource_ids: ["file_a", "file_b"],
      grantee_user_id: "charlie",
      grantee_org_id: "org-c",
      role: "read",
      metadata: { reason: "bulk review" },
    });
    const folderCreated = await client.createResourceCollectionShareGrants("collection_nph", {
      grantee_user_id: "dana",
      grantee_org_id: "org-d",
      role: "read",
      metadata: { reason: "folder review" },
    });

    expect(listed.grants[0].grant_id).toBe("resource_grant_bob");
    expect(created.grant.status).toBe("active");
    expect(revoked.grant.status).toBe("revoked");
    expect(bulkCreated.count).toBe(2);
    expect(bulkCreated.grants.map((grant) => grant.resource_id)).toEqual(["file_a", "file_b"]);
    expect(folderCreated.collection.collection_id).toBe("collection_nph");
    expect(folderCreated.grants.map((grant) => grant.resource_id)).toEqual(["file_a", "file_b"]);
  });

  it("bulk deletes resources through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/resources/delete/bulk") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({
          resource_ids: ["file_a", "file_b"],
        });
        return new Response(
          JSON.stringify({
            count: 2,
            resources: [
              { file_id: "file_a", status: "deleted", original_name: "a.nii" },
              { file_id: "file_b", status: "deleted", original_name: "b.nii" },
            ],
            events: [
              { event_id: "event_a", resource_id: "file_a", event_type: "resource.deleted" },
              { event_id: "event_b", resource_id: "file_b", event_type: "resource.deleted" },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const deleted = await client.deleteResources({ resource_ids: ["file_a", "file_b"] });

    expect(deleted.count).toBe(2);
    expect(deleted.events.map((event) => event.resource_id)).toEqual(["file_a", "file_b"]);
  });

  it("bulk restores resources through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/resources/restore/bulk") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({
          resource_ids: ["file_a", "file_b"],
        });
        return new Response(
          JSON.stringify({
            count: 2,
            resources: [
              { file_id: "file_a", status: "active", original_name: "a.nii" },
              { file_id: "file_b", status: "active", original_name: "b.nii" },
            ],
            events: [
              { event_id: "event_a", resource_id: "file_a", event_type: "resource.restored" },
              { event_id: "event_b", resource_id: "file_b", event_type: "resource.restored" },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const restored = await client.restoreResources({ resource_ids: ["file_a", "file_b"] });

    expect(restored.count).toBe(2);
    expect(restored.resources.map((resource) => resource.status)).toEqual(["active", "active"]);
    expect(restored.events.map((event) => event.event_type)).toEqual([
      "resource.restored",
      "resource.restored",
    ]);
  });

  it("restores a soft-deleted resource through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/resources/file_a/restore") {
        expect(init?.method).toBe("POST");
        return new Response(
          JSON.stringify({
            resource: {
              file_id: "file_a",
              original_name: "a.nii",
              status: "active",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const restored = await client.restoreResource("file_a");

    expect(restored.resource.file_id).toBe("file_a");
    expect(restored.resource.status).toBe("active");
  });

  it("renames resources and removes folder memberships through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/resources/file_a") {
        expect(init?.method).toBe("PATCH");
        expect(JSON.parse(String(init?.body))).toEqual({
          original_name: "nph-a-reviewed.nii.gz",
        });
        return new Response(
          JSON.stringify({
            resource: {
              file_id: "file_a",
              original_name: "nph-a-reviewed.nii.gz",
              status: "active",
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/resource-collections/collection_nph") {
        expect(init?.method).toBe("PATCH");
        expect(JSON.parse(String(init?.body))).toEqual({ name: "NPH reviewed" });
        return new Response(
          JSON.stringify({
            collection: {
              collection_id: "collection_nph",
              owner_user_id: "nph-user",
              name: "NPH reviewed",
              collection_type: "folder",
              status: "active",
              resource_count: 2,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:05:00Z",
              metadata: {},
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (
        url ===
        "https://ultra.example.org/v2/resource-collections/collection_nph/resources/file_a"
      ) {
        expect(init?.method).toBe("DELETE");
        return new Response(
          JSON.stringify({
            collection: {
              collection_id: "collection_nph",
              owner_user_id: "nph-user",
              name: "NPH reviewed",
              collection_type: "folder",
              status: "active",
              resource_count: 1,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:06:00Z",
              metadata: {},
            },
            removed_count: 1,
            memberships: [
              {
                collection_id: "collection_nph",
                resource_id: "file_a",
                position: 0,
                added_at: "2026-06-08T00:00:01Z",
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

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const renamedResource = await client.renameResource(" file_a ", " nph-a-reviewed.nii.gz ");
    const renamedCollection = await client.patchResourceCollection(" collection_nph ", {
      name: " NPH reviewed ",
    });
    const removed = await client.removeResourceFromCollection(" collection_nph ", " file_a ");

    expect(renamedResource.resource.original_name).toBe("nph-a-reviewed.nii.gz");
    expect(renamedCollection.collection.name).toBe("NPH reviewed");
    expect(removed.removed_count).toBe(1);
    expect(removed.collection.resource_count).toBe(1);
  });

  it("manages resource collections through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/resource-collections") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toMatchObject({
          name: "NPH NIfTI cohort",
          collection_type: "folder",
          project_id: "nph-study",
        });
        return new Response(
          JSON.stringify({
            collection: {
              collection_id: "collection_nph",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              project_id: "nph-study",
              name: "NPH NIfTI cohort",
              description: "NIfTI files labeled NPH",
              collection_type: "folder",
              status: "active",
              resource_count: 0,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: { label: "NPH" },
            },
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/resource-collections/collection_nph/resources") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({ resource_ids: ["file_a", "file_b"] });
        return new Response(
          JSON.stringify({
            collection: {
              collection_id: "collection_nph",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              project_id: "nph-study",
              name: "NPH NIfTI cohort",
              collection_type: "folder",
              status: "active",
              resource_count: 2,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:01Z",
              metadata: {},
            },
            added_count: 2,
            memberships: [
              {
                collection_id: "collection_nph",
                resource_id: "file_a",
                position: 0,
                added_at: "2026-06-08T00:00:01Z",
                metadata: {},
              },
              {
                collection_id: "collection_nph",
                resource_id: "file_b",
                position: 1,
                added_at: "2026-06-08T00:00:01Z",
                metadata: {},
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/resource-collections/collection_nph/resources?limit=25&offset=0&q=nph&kind=image&source=upload") {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            count: 2,
            resources: [
              {
                file_id: "file_a",
                original_name: "nph-a.nii.gz",
                content_type: "application/gzip",
                size_bytes: 128,
                sha256: "sha-a",
                created_at: "2026-06-08T00:00:00Z",
                source_type: "upload",
                resource_kind: "image",
                has_thumbnail: false,
              },
              {
                file_id: "file_b",
                original_name: "nph-b.nii.gz",
                content_type: "application/gzip",
                size_bytes: 256,
                sha256: "sha-b",
                created_at: "2026-06-08T00:00:00Z",
                source_type: "upload",
                resource_kind: "image",
                has_thumbnail: false,
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/resource-collections?limit=10&offset=0&collection_type=folder&project_id=nph-study") {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            count: 1,
            collections: [
              {
                collection_id: "collection_nph",
                owner_user_id: "nph-user",
                owner_org_id: "nph-org",
                project_id: "nph-study",
                name: "NPH NIfTI cohort",
                collection_type: "folder",
                status: "active",
                resource_count: 2,
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:01Z",
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

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const created = await client.createResourceCollection({
      name: "NPH NIfTI cohort",
      description: "NIfTI files labeled NPH",
      collection_type: "folder",
      project_id: "nph-study",
      metadata: { label: "NPH" },
    });
    const added = await client.addResourcesToCollection(created.collection.collection_id, [
      "file_a",
      "file_b",
    ]);
    const resources = await client.listResourceCollectionResources(created.collection.collection_id, {
      limit: 25,
      query: "nph",
      kind: "image",
      source: "upload",
    });
    const collections = await client.listResourceCollections({
      limit: 10,
      collectionType: "folder",
      projectId: "nph-study",
    });

    expect(created.collection.collection_id).toBe("collection_nph");
    expect(added.added_count).toBe(2);
    expect(resources.resources.map((resource) => resource.file_id)).toEqual(["file_a", "file_b"]);
    expect(collections.collections[0].resource_count).toBe(2);
  });

  it("lists deleted resource collections and restores a folder through V2", async () => {
    const seenRequests: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      seenRequests.push(`${init?.method ?? "GET"} ${url}`);
      if (url === "https://ultra.example.org/v2/resource-collections?limit=25&offset=0&collection_type=folder&status=deleted") {
        return new Response(
          JSON.stringify({
            count: 1,
            collections: [
              {
                collection_id: "collection_deleted_nph",
                owner_user_id: "nph-user",
                owner_org_id: "nph-org",
                name: "Deleted NPH review folder",
                collection_type: "folder",
                status: "deleted",
                resource_count: 2,
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:05:00Z",
                metadata: {},
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/resource-collections/collection_deleted_nph/restore") {
        expect(init?.method).toBe("POST");
        return new Response(
          JSON.stringify({
            collection: {
              collection_id: "collection_deleted_nph",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              name: "Deleted NPH review folder",
              collection_type: "folder",
              status: "active",
              resource_count: 2,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:06:00Z",
              metadata: {},
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/resource-collections/collection_deleted_nph") {
        expect(init?.method).toBe("DELETE");
        return new Response(
          JSON.stringify({
            collection: {
              collection_id: "collection_deleted_nph",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              name: "Deleted NPH review folder",
              collection_type: "folder",
              status: "deleted",
              resource_count: 2,
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:05:00Z",
              metadata: {},
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const deleted = await client.listResourceCollections({
      limit: 25,
      collectionType: "folder",
      status: "deleted",
    });
    const restored = await client.restoreResourceCollection("collection_deleted_nph");
    const deletedAgain = await client.deleteResourceCollection("collection_deleted_nph");

    expect(deleted.collections[0].status).toBe("deleted");
    expect(restored.collection.status).toBe("active");
    expect(deletedAgain.collection.status).toBe("deleted");
    expect(seenRequests).toEqual([
      "GET https://ultra.example.org/v2/resource-collections?limit=25&offset=0&collection_type=folder&status=deleted",
      "POST https://ultra.example.org/v2/resource-collections/collection_deleted_nph/restore",
      "DELETE https://ultra.example.org/v2/resource-collections/collection_deleted_nph",
    ]);
  });

  it("creates and loads immutable dataset snapshots through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/dataset-snapshots") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toMatchObject({
          name: "NPH training cohort v1",
          source_collection_id: "collection_nph",
        });
        return new Response(
          JSON.stringify({
            snapshot: {
              snapshot_id: "dataset_snapshot_nph_v1",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              source_collection_id: "collection_nph",
              name: "NPH training cohort v1",
              description: "Frozen folder manifest for training",
              status: "active",
              resource_count: 2,
              total_bytes: 384,
              created_by_user_id: "nph-user",
              created_at: "2026-06-08T00:00:00Z",
              metadata: { label: "NPH" },
            },
            resources: [
              {
                snapshot_id: "dataset_snapshot_nph_v1",
                resource_id: "file_a",
                position: 0,
                original_name: "nph-a.nii.gz",
                resource_kind: "file",
                source_type: "upload",
                size_bytes: 128,
                sha256: "sha-a",
              },
              {
                snapshot_id: "dataset_snapshot_nph_v1",
                resource_id: "file_b",
                position: 1,
                original_name: "nph-b.nii.gz",
                resource_kind: "file",
                source_type: "upload",
                size_bytes: 256,
                sha256: "sha-b",
              },
            ],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      if (
        url ===
        "https://ultra.example.org/v2/dataset-snapshots?limit=10&offset=0&q=training&project_id=nph-study&source_collection_id=collection_nph"
      ) {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            count: 1,
            snapshots: [
              {
                snapshot_id: "dataset_snapshot_nph_v1",
                owner_user_id: "nph-user",
                owner_org_id: "nph-org",
                project_id: "nph-study",
                source_collection_id: "collection_nph",
                name: "NPH training cohort v1",
                status: "active",
                resource_count: 2,
                total_bytes: 384,
                created_by_user_id: "nph-user",
                created_at: "2026-06-08T00:00:00Z",
                metadata: { label: "NPH" },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/dataset-snapshots/dataset_snapshot_nph_v1") {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            snapshot: {
              snapshot_id: "dataset_snapshot_nph_v1",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              source_collection_id: "collection_nph",
              name: "NPH training cohort v1",
              status: "active",
              resource_count: 2,
              total_bytes: 384,
              created_by_user_id: "nph-user",
              created_at: "2026-06-08T00:00:00Z",
              metadata: { label: "NPH" },
            },
            resources: [
              {
                snapshot_id: "dataset_snapshot_nph_v1",
                resource_id: "file_a",
                position: 0,
                original_name: "nph-a.nii.gz",
                resource_kind: "file",
                source_type: "upload",
                size_bytes: 128,
                sha256: "sha-a",
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
    const created = await client.createDatasetSnapshot({
      name: "NPH training cohort v1",
      description: "Frozen folder manifest for training",
      source_collection_id: "collection_nph",
      metadata: { label: "NPH" },
    });
    const listed = await client.listDatasetSnapshots({
      limit: 10,
      query: "training",
      projectId: "nph-study",
      sourceCollectionId: "collection_nph",
    });
    const loaded = await client.getDatasetSnapshot(created.snapshot.snapshot_id);

    expect(created.snapshot.resource_count).toBe(2);
    expect(created.resources.map((resource) => resource.sha256)).toEqual(["sha-a", "sha-b"]);
    expect(listed.snapshots.map((snapshot) => snapshot.snapshot_id)).toEqual([
      "dataset_snapshot_nph_v1",
    ]);
    expect(loaded.snapshot.snapshot_id).toBe("dataset_snapshot_nph_v1");
    expect(loaded.resources[0].resource_id).toBe("file_a");
  });

  it("lists deleted dataset snapshots and restores them through V2", async () => {
    const seenRequests: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      seenRequests.push(`${init?.method ?? "GET"} ${url}`);
      if (
        url ===
        "https://ultra.example.org/v2/dataset-snapshots?limit=25&offset=0&project_id=nph-study&status=deleted"
      ) {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            count: 1,
            snapshots: [
              {
                snapshot_id: "dataset_snapshot_deleted_nph",
                owner_user_id: "nph-user",
                owner_org_id: "nph-org",
                project_id: "nph-study",
                source_collection_id: "collection_nph",
                name: "Deleted NPH training cohort",
                status: "deleted",
                resource_count: 2,
                total_bytes: 384,
                created_by_user_id: "nph-user",
                created_at: "2026-06-08T00:00:00Z",
                metadata: {},
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (
        url ===
        "https://ultra.example.org/v2/dataset-snapshots/dataset_snapshot_deleted_nph/restore"
      ) {
        expect(init?.method).toBe("POST");
        return new Response(
          JSON.stringify({
            snapshot: {
              snapshot_id: "dataset_snapshot_deleted_nph",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              project_id: "nph-study",
              source_collection_id: "collection_nph",
              name: "Deleted NPH training cohort",
              status: "active",
              resource_count: 2,
              total_bytes: 384,
              created_by_user_id: "nph-user",
              created_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            resources: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/dataset-snapshots/dataset_snapshot_deleted_nph") {
        expect(init?.method).toBe("DELETE");
        return new Response(
          JSON.stringify({
            snapshot: {
              snapshot_id: "dataset_snapshot_deleted_nph",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              project_id: "nph-study",
              source_collection_id: "collection_nph",
              name: "Deleted NPH training cohort",
              status: "deleted",
              resource_count: 2,
              total_bytes: 384,
              created_by_user_id: "nph-user",
              created_at: "2026-06-08T00:00:00Z",
              metadata: {},
            },
            resources: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const deleted = await client.listDatasetSnapshots({
      limit: 25,
      projectId: "nph-study",
      status: "deleted",
    });
    const restored = await client.restoreDatasetSnapshot("dataset_snapshot_deleted_nph");
    const deletedAgain = await client.deleteDatasetSnapshot("dataset_snapshot_deleted_nph");

    expect(deleted.snapshots[0].status).toBe("deleted");
    expect(restored.snapshot.status).toBe("active");
    expect(deletedAgain.snapshot.status).toBe("deleted");
    expect(seenRequests).toEqual([
      "GET https://ultra.example.org/v2/dataset-snapshots?limit=25&offset=0&project_id=nph-study&status=deleted",
      "POST https://ultra.example.org/v2/dataset-snapshots/dataset_snapshot_deleted_nph/restore",
      "DELETE https://ultra.example.org/v2/dataset-snapshots/dataset_snapshot_deleted_nph",
    ]);
  });

  it("creates immutable dataset snapshots from resource query filters through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/dataset-snapshots") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({
          name: "NPH under 70 query cohort",
          resource_query: {
            q: "NPH",
            kind: "file",
            source: "upload",
            sharing: "private",
            tags: ["Under 70"],
          },
          metadata: {
            source: "resources_query_toolbar",
            query_result_count: 2,
          },
        });
        return new Response(
          JSON.stringify({
            snapshot: {
              snapshot_id: "dataset_snapshot_query_under70",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              name: "NPH under 70 query cohort",
              status: "active",
              resource_count: 2,
              total_bytes: 384,
              created_by_user_id: "nph-user",
              created_at: "2026-06-08T00:00:00Z",
              metadata: { source: "resources_query_toolbar" },
            },
            resources: [
              {
                snapshot_id: "dataset_snapshot_query_under70",
                resource_id: "file_query_dataset_b",
                position: 0,
                original_name: "subject-b-nph-under70.nii.gz",
                resource_kind: "file",
                source_type: "upload",
                size_bytes: 256,
                sha256: "sha-query-b",
              },
              {
                snapshot_id: "dataset_snapshot_query_under70",
                resource_id: "file_query_dataset_a",
                position: 1,
                original_name: "subject-a-nph-under70.nii.gz",
                resource_kind: "file",
                source_type: "upload",
                size_bytes: 128,
                sha256: "sha-query-a",
              },
            ],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const created = await client.createDatasetSnapshot({
      name: "NPH under 70 query cohort",
      resource_query: {
        q: "NPH",
        kind: "file",
        source: "upload",
        sharing: "private",
        tags: ["Under 70"],
      },
      metadata: {
        source: "resources_query_toolbar",
        query_result_count: 2,
      },
    });

    expect(created.snapshot.snapshot_id).toBe("dataset_snapshot_query_under70");
    expect(created.resources.map((resource) => resource.resource_id)).toEqual([
      "file_query_dataset_b",
      "file_query_dataset_a",
    ]);
  });

  it("manages dataset snapshot share grants through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (
        url ===
        "https://ultra.example.org/v2/dataset-snapshots/dataset_snapshot_nph_v1/shares?limit=25&status=active"
      ) {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            count: 1,
            grants: [
              {
                grant_id: "dataset_snapshot_grant_bob",
                snapshot_id: "dataset_snapshot_nph_v1",
                owner_user_id: "alice",
                owner_org_id: "org-a",
                grantee_user_id: "bob",
                grantee_org_id: "org-b",
                role: "read",
                status: "active",
                created_by_user_id: "alice",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:00Z",
                metadata: { reason: "review cohort" },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (
        url === "https://ultra.example.org/v2/dataset-snapshots/dataset_snapshot_nph_v1/shares" &&
        init?.method === "POST"
      ) {
        expect(JSON.parse(String(init.body))).toEqual({
          grantee_user_id: "bob",
          grantee_org_id: "org-b",
          role: "read",
          metadata: { source: "resources_dataset_share_panel" },
        });
        return new Response(
          JSON.stringify({
            grant: {
              grant_id: "dataset_snapshot_grant_bob",
              snapshot_id: "dataset_snapshot_nph_v1",
              owner_user_id: "alice",
              owner_org_id: "org-a",
              grantee_user_id: "bob",
              grantee_org_id: "org-b",
              role: "read",
              status: "active",
              created_by_user_id: "alice",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              metadata: { source: "resources_dataset_share_panel" },
            },
          }),
          { status: 201, headers: { "Content-Type": "application/json" } }
        );
      }
      if (
        url ===
        "https://ultra.example.org/v2/dataset-snapshots/dataset_snapshot_nph_v1/shares/dataset_snapshot_grant_bob"
      ) {
        expect(init?.method).toBe("DELETE");
        return new Response(
          JSON.stringify({
            grant: {
              grant_id: "dataset_snapshot_grant_bob",
              snapshot_id: "dataset_snapshot_nph_v1",
              owner_user_id: "alice",
              owner_org_id: "org-a",
              grantee_user_id: "bob",
              grantee_org_id: "org-b",
              role: "read",
              status: "revoked",
              created_by_user_id: "alice",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:01:00Z",
              revoked_at: "2026-06-08T00:01:00Z",
              metadata: { source: "resources_dataset_share_panel" },
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const listed = await client.listDatasetSnapshotShareGrants("dataset_snapshot_nph_v1", {
      limit: 25,
      status: "active",
    });
    const created = await client.createDatasetSnapshotShareGrant("dataset_snapshot_nph_v1", {
      grantee_user_id: "bob",
      grantee_org_id: "org-b",
      role: "read",
      metadata: { source: "resources_dataset_share_panel" },
    });
    const revoked = await client.revokeDatasetSnapshotShareGrant(
      "dataset_snapshot_nph_v1",
      created.grant.grant_id
    );

    expect(listed.grants).toHaveLength(1);
    expect(created.grant.grantee_user_id).toBe("bob");
    expect(revoked.grant.status).toBe("revoked");
  });

  it("lists dataset snapshot audit events through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (
        url ===
        "https://ultra.example.org/v2/dataset-snapshots/dataset_snapshot_nph_v1/events?limit=25&offset=0&event_type=dataset_snapshot.shared&actor_user_id=alice"
      ) {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            snapshot_id: "dataset_snapshot_nph_v1",
            count: 1,
            total_count: 1,
            limit: 25,
            offset: 0,
            events: [
              {
                event_id: "dataset_snapshot_event_shared",
                snapshot_id: "dataset_snapshot_nph_v1",
                actor_user_id: "alice",
                actor_org_id: "org-a",
                event_type: "dataset_snapshot.shared",
                ts: "2026-06-08T00:01:00Z",
                metadata: { grant_id: "dataset_snapshot_grant_bob" },
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
    const listed = await client.listDatasetSnapshotEvents("dataset_snapshot_nph_v1", {
      limit: 25,
      offset: 0,
      eventType: "dataset_snapshot.shared",
      actorUserId: "alice",
    });

    expect(listed.snapshot_id).toBe("dataset_snapshot_nph_v1");
    expect(listed.events).toHaveLength(1);
    expect(listed.events[0].event_type).toBe("dataset_snapshot.shared");
    expect(listed.events[0].metadata?.grant_id).toBe("dataset_snapshot_grant_bob");
  });

  it("creates lists and loads durable data agent jobs through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/data-agent/jobs") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toMatchObject({
          job_type: "caption_resources",
          resource_ids: ["file_a", "file_b"],
          input_selector: { mode: "short_caption", label: "NPH" },
        });
        return new Response(
          JSON.stringify({
            job: {
              job_id: "data_agent_job_nph_caption",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              project_id: "nph-study",
              job_type: "caption_resources",
              status: "queued",
              resource_count: 2,
              progress_completed: 0,
              progress_total: 2,
              created_by_user_id: "nph-user",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              input_selector: { mode: "short_caption", label: "NPH" },
              output_summary: {},
              metadata: { requested_from: "resources_page" },
            },
            events: [
              {
                event_id: "data_agent_job_event_created",
                job_id: "data_agent_job_nph_caption",
                sequence: 1,
                event_type: "data_agent.job.created",
                actor_user_id: "nph-user",
                actor_org_id: "nph-org",
                ts: "2026-06-08T00:00:00Z",
                message: "Data Agent job queued.",
                metadata: { resource_count: 2 },
              },
            ],
          }),
          { status: 202, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/data-agent/jobs?limit=10&offset=0&status=queued&job_type=caption_resources") {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            count: 1,
            jobs: [
              {
                job_id: "data_agent_job_nph_caption",
                owner_user_id: "nph-user",
                owner_org_id: "nph-org",
                project_id: "nph-study",
                job_type: "caption_resources",
                status: "queued",
                resource_count: 2,
                progress_completed: 0,
                progress_total: 2,
                created_by_user_id: "nph-user",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:00:00Z",
                input_selector: { mode: "short_caption", label: "NPH" },
                output_summary: {},
                metadata: { requested_from: "resources_page" },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/data-agent/jobs/data_agent_job_nph_caption") {
        expect(init?.method).toBe("GET");
        return new Response(
          JSON.stringify({
            job: {
              job_id: "data_agent_job_nph_caption",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              project_id: "nph-study",
              job_type: "caption_resources",
              status: "queued",
              resource_count: 2,
              progress_completed: 0,
              progress_total: 2,
              created_by_user_id: "nph-user",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:00:00Z",
              input_selector: { mode: "short_caption", label: "NPH" },
              output_summary: {},
              metadata: { requested_from: "resources_page" },
            },
            events: [
              {
                event_id: "data_agent_job_event_created",
                job_id: "data_agent_job_nph_caption",
                sequence: 1,
                event_type: "data_agent.job.created",
                actor_user_id: "nph-user",
                actor_org_id: "nph-org",
                ts: "2026-06-08T00:00:00Z",
                message: "Data Agent job queued.",
                metadata: { resource_count: 2 },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/data-agent/jobs/data_agent_job_nph_caption/status") {
        expect(init?.method).toBe("PATCH");
        expect(JSON.parse(String(init?.body))).toMatchObject({
          status: "running",
          progress_completed: 1,
          progress_total: 2,
          message: "Captioned first resource",
        });
        return new Response(
          JSON.stringify({
            job: {
              job_id: "data_agent_job_nph_caption",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              project_id: "nph-study",
              job_type: "caption_resources",
              status: "running",
              resource_count: 2,
              progress_completed: 1,
              progress_total: 2,
              created_by_user_id: "nph-user",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:01:00Z",
              started_at: "2026-06-08T00:01:00Z",
              input_selector: { mode: "short_caption", label: "NPH" },
              output_summary: {},
              metadata: { requested_from: "resources_page" },
            },
            events: [
              {
                event_id: "data_agent_job_event_created",
                job_id: "data_agent_job_nph_caption",
                sequence: 1,
                event_type: "data_agent.job.created",
                ts: "2026-06-08T00:00:00Z",
                metadata: { resource_count: 2 },
              },
              {
                event_id: "data_agent_job_event_progressed",
                job_id: "data_agent_job_nph_caption",
                sequence: 2,
                event_type: "data_agent.job.progressed",
                ts: "2026-06-08T00:01:00Z",
                message: "Captioned first resource",
                metadata: { resource_id: "file_a" },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/data-agent/jobs/data_agent_job_nph_caption/control") {
        expect(init?.method).toBe("POST");
        const body = JSON.parse(String(init?.body));
        if (body.action === "cancel") {
          return new Response(
            JSON.stringify({
              job: {
                job_id: "data_agent_job_nph_caption",
                owner_user_id: "nph-user",
                owner_org_id: "nph-org",
                project_id: "nph-study",
                job_type: "caption_resources",
                status: "canceled",
                resource_count: 2,
                progress_completed: 1,
                progress_total: 2,
                error: "User paused the field upload.",
                created_by_user_id: "nph-user",
                created_at: "2026-06-08T00:00:00Z",
                updated_at: "2026-06-08T00:02:00Z",
                started_at: "2026-06-08T00:01:00Z",
                completed_at: "2026-06-08T00:02:00Z",
                input_selector: { mode: "short_caption", label: "NPH" },
                output_summary: {},
                metadata: { requested_from: "resources_page" },
              },
              events: [
                {
                  event_id: "data_agent_job_event_canceled",
                  job_id: "data_agent_job_nph_caption",
                  sequence: 3,
                  event_type: "data_agent.job.canceled",
                  ts: "2026-06-08T00:02:00Z",
                  message: "User paused the field upload.",
                  metadata: {},
                },
              ],
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        expect(body).toMatchObject({ action: "retry", reason: "Connectivity recovered." });
        return new Response(
          JSON.stringify({
            job: {
              job_id: "data_agent_job_nph_caption",
              owner_user_id: "nph-user",
              owner_org_id: "nph-org",
              project_id: "nph-study",
              job_type: "caption_resources",
              status: "queued",
              resource_count: 2,
              progress_completed: 0,
              progress_total: 2,
              created_by_user_id: "nph-user",
              created_at: "2026-06-08T00:00:00Z",
              updated_at: "2026-06-08T00:03:00Z",
              input_selector: { mode: "short_caption", label: "NPH" },
              output_summary: {},
              metadata: { requested_from: "resources_page" },
            },
            events: [
              {
                event_id: "data_agent_job_event_retried",
                job_id: "data_agent_job_nph_caption",
                sequence: 4,
                event_type: "data_agent.job.retried",
                ts: "2026-06-08T00:03:00Z",
                message: "Connectivity recovered.",
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

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const created = await client.createDataAgentJob({
      job_type: "caption_resources",
      resource_ids: ["file_a", "file_b"],
      project_id: "nph-study",
      input_selector: { mode: "short_caption", label: "NPH" },
      metadata: { requested_from: "resources_page" },
    });
    const listed = await client.listDataAgentJobs({
      limit: 10,
      status: "queued",
      jobType: "caption_resources",
    });
    const loaded = await client.getDataAgentJob(created.job.job_id);
    const progressed = await client.updateDataAgentJobStatus(created.job.job_id, {
      status: "running",
      progress_completed: 1,
      progress_total: 2,
      message: "Captioned first resource",
      event_metadata: { resource_id: "file_a" },
    });
    const canceled = await client.controlDataAgentJob(created.job.job_id, {
      action: "cancel",
      reason: "User paused the field upload.",
    });
    const retried = await client.controlDataAgentJob(created.job.job_id, {
      action: "retry",
      reason: "Connectivity recovered.",
    });

    expect(created.job.status).toBe("queued");
    expect(created.events[0].event_type).toBe("data_agent.job.created");
    expect(listed.jobs[0].job_id).toBe("data_agent_job_nph_caption");
    expect(loaded.job.resource_count).toBe(2);
    expect(progressed.job.status).toBe("running");
    expect(progressed.events[1].event_type).toBe("data_agent.job.progressed");
    expect(canceled.job.status).toBe("canceled");
    expect(retried.job.status).toBe("queued");
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

  it("recovers when a cached V2 thread id is invalid or corrupt", async () => {
    browserStorage().setItem(
      "bisque-ultra:v2-chat-thread-map",
      JSON.stringify({
        "conversation-local-123": "0013d3a4-aa72-4da1-99e5-5613cb164e17",
      })
    );

    const encoder = new TextEncoder();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
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
      messages: [{ role: "user", content: "What are my most recent uploads to BisQue?" }],
      uploaded_files: [],
      conversation_id: "conversation-local-123",
      goal: "What are my most recent uploads to BisQue?",
    });

    expect(response.run_id).toBe("run_fresh");
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
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
            total_count: 40,
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
      total_count: 40,
      limit: 25,
      offset: 0,
      has_more: true,
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

  it("derives default V2 sidebar titles from durable thread messages", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/threads?limit=25&offset=0") {
        return new Response(
          JSON.stringify({
            count: 1,
            total_count: 1,
            limit: 25,
            offset: 0,
            has_more: false,
            threads: [
              {
                thread_id: "thread_v2_default_title",
                title: "New conversation",
                status: "active",
                created_at: "2026-06-03T10:00:00Z",
                updated_at: "2026-06-03T10:01:00Z",
                latest_run_id: "run_v2_default_title",
                metadata: {
                  conversation_id: "conversation-local-default-title",
                },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/threads/thread_v2_default_title/messages") {
        return new Response(
          JSON.stringify({
            thread_id: "thread_v2_default_title",
            count: 2,
            messages: [
              {
                role: "user",
                content: "Run prairie dog detection on this image and discuss the results.",
                created_at: "2026-06-03T10:00:00Z",
                run_id: "run_v2_default_title",
              },
              {
                role: "assistant",
                content: "RareSpot completed.",
                created_at: "2026-06-03T10:01:00Z",
                run_id: "run_v2_default_title",
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/runs/run_v2_default_title") {
        return new Response(
          JSON.stringify({
            run_id: "run_v2_default_title",
            thread_id: "thread_v2_default_title",
            goal: "Run prairie dog detection on this image and discuss the results.",
            status: "succeeded",
            workflow_kind: "deepagents",
            response_text: "RareSpot completed.",
            created_at: "2026-06-03T10:00:00Z",
            updated_at: "2026-06-03T10:01:00Z",
            metadata: {},
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.listConversations(25, 0, false);

    expect(response.conversations).toMatchObject([
      {
        conversation_id: "conversation-local-default-title",
        title: "Prairie Dog Detection Image Discuss Results",
        preview: "Run prairie dog detection on this image and discuss the results.",
        message_count: 2,
      },
    ]);
    expect(response.conversations[0].state).toEqual({});
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads?limit=25&offset=0",
      "https://ultra.example.org/v2/threads/thread_v2_default_title/messages",
      "https://ultra.example.org/v2/runs/run_v2_default_title",
    ]);
  });

  it("omits empty default V2 threads while preserving page progress", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url === "https://ultra.example.org/v2/threads?limit=25&offset=0") {
        return new Response(
          JSON.stringify({
            count: 1,
            total_count: 1,
            limit: 25,
            offset: 0,
            has_more: false,
            threads: [
              {
                thread_id: "thread_v2_empty_default",
                title: "New conversation",
                status: "active",
                created_at: "2026-06-03T10:00:00Z",
                updated_at: "2026-06-03T10:01:00Z",
                latest_run_id: null,
                metadata: {
                  conversation_id: "conversation-local-empty-default",
                  preview: "",
                  message_count: 0,
                },
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/threads/thread_v2_empty_default/messages") {
        return new Response(
          JSON.stringify({
            thread_id: "thread_v2_empty_default",
            count: 0,
            messages: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.listConversations(25, 0, false);

    expect(response).toMatchObject({
      count: 1,
      total_count: 1,
      limit: 25,
      offset: 0,
      has_more: false,
      conversations: [],
    });
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads?limit=25&offset=0",
      "https://ultra.example.org/v2/threads/thread_v2_empty_default/messages",
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

  it("marks explicit sidebar renames as manual title updates", async () => {
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
            title: "Manual ecology review",
            status: "active",
            created_at: "2026-05-31T11:16:00Z",
            updated_at: "2026-05-31T11:17:00Z",
            metadata: {
              conversation_id: "conversation-local-123",
              title_state: {
                source: "manual",
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
    await client.upsertConversation(
      {
        conversation_id: "conversation-local-123",
        title: "Manual ecology review",
        created_at_ms: Date.parse("2026-05-31T11:16:00Z"),
        updated_at_ms: Date.parse("2026-05-31T11:17:00Z"),
        preview: "Run RareSpot analysis on prairie dog imagery",
        message_count: 2,
        preferred_panel: "chat",
        running: false,
        state: {},
      },
      { titleSource: "manual" }
    );

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toMatchObject({
      title: "Manual ecology review",
      metadata: {
        conversation_id: "conversation-local-123",
        title_state: {
          source: "manual",
        },
      },
    });
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

  it("reconciles stale frontend state with a completed durable latest run after refresh", async () => {
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
            title: "run durable analysis",
            status: "active",
            created_at: "2026-05-31T11:16:00Z",
            updated_at: "2026-05-31T11:17:00Z",
            latest_run_id: "run_v2_123",
            metadata: {
              conversation_id: "conversation-local-123",
              frontend_state: {
                sending: true,
                streamingMessageId: "msg-assistant",
                messages: [
                  {
                    id: "msg-user",
                    role: "user",
                    content: "run durable analysis",
                    createdAt: Date.parse("2026-05-31T11:16:00Z"),
                  },
                  {
                    id: "msg-assistant",
                    role: "assistant",
                    content: "",
                    createdAt: Date.parse("2026-05-31T11:16:01Z"),
                  },
                ],
              },
            },
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
                message_id: "msg_durable_user",
                role: "user",
                content: "run durable analysis",
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
            response_text: "Durable answer restored.",
            completed_at: "2026-05-31T11:17:00Z",
            updated_at: "2026-05-31T11:17:00Z",
            metadata: { response_layout: { sections: [] } },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.getConversation("conversation-local-123");

    expect(response.running).toBe(false);
    expect(response.message_count).toBe(2);
    expect(response.state).toMatchObject({
      sending: false,
      streamingMessageId: null,
      messages: [
        {
          id: "msg-user",
          role: "user",
          content: "run durable analysis",
        },
        {
          id: "msg-assistant",
          role: "assistant",
          content: "Durable answer restored.",
          runId: "run_v2_123",
          responseMetadata: { response_layout: { sections: [] } },
        },
      ],
    });
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads/thread_v2_123",
      "https://ultra.example.org/v2/threads/thread_v2_123/messages",
      "https://ultra.example.org/v2/runs/run_v2_123",
    ]);
  });

  it("patches stale frontend state with an active latest run id for stream recovery", async () => {
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
            title: "run long analysis",
            status: "active",
            created_at: "2026-05-31T11:16:00Z",
            updated_at: "2026-05-31T11:17:00Z",
            latest_run_id: "run_v2_123",
            metadata: {
              conversation_id: "conversation-local-123",
              frontend_state: {
                sending: true,
                streamingMessageId: "msg-assistant",
                messages: [
                  {
                    id: "msg-user",
                    role: "user",
                    content: "run long analysis",
                  },
                  {
                    id: "msg-assistant",
                    role: "assistant",
                    content: "",
                  },
                ],
              },
            },
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
                message_id: "msg_durable_user",
                role: "user",
                content: "run long analysis",
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
            response_text: "",
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

    expect(response.running).toBe(true);
    expect(response.state).toMatchObject({
      sending: true,
      streamingMessageId: "msg-assistant",
      messages: [
        {
          id: "msg-user",
          role: "user",
          content: "run long analysis",
        },
        {
          id: "msg-assistant",
          role: "assistant",
          content: "",
          runId: "run_v2_123",
        },
      ],
    });
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
      "https://ultra.example.org/v2/threads/thread_v2_123",
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
      if (url === "https://ultra.example.org/v2/auth/request-account") {
        return new Response(JSON.stringify({ authenticated: false, account_status: "pending" }), {
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
      if (url === "https://ultra.example.org/v2/bisque/unlink") {
        return new Response(JSON.stringify({ authenticated: false, user: null, bisque_linked: false }), {
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
    await client.requestAccount({
      name: "Grace",
      email: "grace@example.org",
      affiliation: "US Navy",
    });
    await client.startHostedAuth();
    await client.loginBisque({ username: "local", password: "local" });
    await client.logoutBisque();
    await client.unlinkBisqueAccount();

    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls.every((url) => url.includes("/v2/"))).toBe(true);
    expect(urls.some((url) => url.includes("/v1/"))).toBe(false);
    expect(urls).toContain("https://ultra.example.org/v2/auth/request-account");
    expect(urls).not.toContain("https://ultra.example.org/v2/auth/guest");
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
    expect(viewer.service_urls?.tile).toBe("/v2/uploads/file-1/tiles");
    expect(viewer.service_urls?.atlas).toBe("/v2/uploads/file-1/atlas");
    expect(viewer.service_urls?.histogram).toBe("/v2/uploads/file-1/histogram");
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

  it("searches BisQue resources through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/bisque/search") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({
          resource_type: "image",
          tag_query: "species:prairie_dog",
          query: "",
          limit: 5,
        });
        return new Response(
          JSON.stringify({
            count: 1,
            results: [
              {
                resource_uri: "https://bisque.example.org/data_service/image/1",
                name: "prairie.jpg",
                resource_type: "image",
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
    const response = await client.searchBisqueResources({
      resourceType: "image",
      tagQuery: "species:prairie_dog",
      limit: 5,
    });

    expect(response.count).toBe(1);
    expect(response.results[0]?.name).toBe("prairie.jpg");
  });

  it("can request BisQue inventory counts through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/bisque/search") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({
          resource_type: "image",
          tag_query: "",
          query: "",
          limit: 1,
          count_all: true,
        });
        return new Response(JSON.stringify({ count: 18, results: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    const response = await client.searchBisqueResources({
      resourceType: "image",
      limit: 1,
      countAll: true,
    });

    expect(response.count).toBe(18);
  });

  it("sends precise BisQue search filters through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/bisque/search") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({
          resource_type: "image",
          tag_query: "",
          tag_order: "@ts:desc",
          query: "",
          name_contains: "EnrNE_",
          extensions: ["png"],
          scope: "owner",
          sort: "recent",
          limit: 10,
          count_all: true,
        });
        return new Response(
          JSON.stringify({
            count: 1,
            results: [
              {
                resource_uri: "https://bisque.example.org/data_service/image/1",
                name: "EnrNE_recent.png",
                resource_type: "image",
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
    const response = await client.searchBisqueResources({
      resourceType: "image",
      tagOrder: "@ts:desc",
      nameContains: "EnrNE_",
      extensions: ["png"],
      scope: "owner",
      sort: "recent",
      limit: 10,
      countAll: true,
    });

    expect(response.count).toBe(1);
    expect(response.results[0]?.name).toBe("EnrNE_recent.png");
  });

  it("downloads and uploads BisQue resources through V2", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "https://ultra.example.org/v2/bisque/download") {
        expect(JSON.parse(String(init?.body))).toEqual({
          resources: ["https://bisque.example.org/data_service/image/1"],
        });
        return new Response(
          JSON.stringify({ file_count: 1, uploaded: [], imports: [] }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url === "https://ultra.example.org/v2/bisque/upload") {
        expect(JSON.parse(String(init?.body))).toEqual({
          file_ids: ["file_1"],
          artifact_ids: ["artifact_1"],
        });
        return new Response(
          JSON.stringify({
            count: 1,
            uploads: [
              {
                file_id: "file_1",
                artifact_id: "artifact_1",
                resource_uri: "https://bisque.example.org/data_service/file/2",
                name: "report.md",
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
    const download = await client.downloadBisqueResources([
      "https://bisque.example.org/data_service/image/1",
    ]);
    const upload = await client.uploadBisqueOutputs({
      fileIds: ["file_1"],
      artifactIds: ["artifact_1"],
    });

    expect(download.file_count).toBe(1);
    expect(upload.count).toBe(1);
    expect(upload.uploads[0]?.resource_uri).toContain("/data_service/file/2");
    expect(fetchMock).toHaveBeenCalledTimes(2);
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

  it("updates admin user approval status through V2", async () => {
    const urls: string[] = [];
    const methods: string[] = [];
    const bodies: string[] = [];
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        urls.push(url);
        methods.push(String(init?.method ?? "GET"));
        bodies.push(String(init?.body ?? ""));
        if (url === "https://ultra.example.org/v2/admin/users/workos%3Auser_pending/status") {
          return new Response(
            JSON.stringify({
              user_id: "workos:user_pending",
              email: "pending@example.org",
              display_name: "Pending Scientist",
              role: "researcher",
              status: "active",
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

    const user = await client.updateAdminUserStatus("workos:user_pending", "active");

    expect(user.status).toBe("active");
    expect(urls).toEqual([
      "https://ultra.example.org/v2/admin/users/workos%3Auser_pending/status",
    ]);
    expect(methods).toEqual(["PATCH"]);
    expect(JSON.parse(bodies[0])).toEqual({ status: "active" });
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

  it("cancels a user's own run through the V2 run cancel endpoint", async () => {
    const urls: string[] = [];
    const methods: string[] = [];
    const bodies: string[] = [];
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        urls.push(url);
        methods.push(String(init?.method ?? "GET"));
        bodies.push(String(init?.body ?? ""));
        if (url === "https://ultra.example.org/v2/runs/run_42/cancel") {
          return new Response(
            JSON.stringify({ run_id: "run_42", status: "canceled" }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response("not found", { status: 404 });
      }
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    await client.cancelRun("run_42", "Stopped from chat composer");

    expect(urls).toEqual(["https://ultra.example.org/v2/runs/run_42/cancel"]);
    expect(methods).toEqual(["POST"]);
    expect(JSON.parse(bodies[0])).toEqual({ reason: "Stopped from chat composer" });
  });

  it("skips the cancel request when no run id is provided", async () => {
    const fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    await client.cancelRun("   ");

    expect(fetchMock).not.toHaveBeenCalled();
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

describe("ApiClient viewer request timeouts", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("times out a hung histogram request as a 504 instead of spinning forever", async () => {
    // A fetch that never resolves until its AbortSignal fires — i.e. a hung image service.
    const fetchMock = vi.fn(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        })
    );
    vi.stubGlobal("fetch", fetchMock);
    vi.useFakeTimers();
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });

    const pending = client.getUploadHistogram("file_x");
    const expectation = expect(pending).rejects.toMatchObject({ status: 504 });
    await vi.advanceTimersByTimeAsync(30_000);
    await expectation;
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("resolves normally when the server responds before the timeout", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ channels: [], bins: 256 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);
    const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
    await expect(client.getUploadHistogram("file_x")).resolves.toMatchObject({ bins: 256 });
  });
});
