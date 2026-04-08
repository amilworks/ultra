import type {
  V2ArtifactListResponse,
  V2ArtifactRecord,
  V2ArtifactResponse,
  V2GraphEventRecord,
  V2RunCancelRequest,
  V2RunCreateRequest,
  V2RunEventsResponse,
  V2RunRecord,
  V2RunResumeRequest,
  V2StreamDoneEvent,
  V2StreamErrorEvent,
  V2StreamEvent,
  V2StreamHandlers,
  V2StreamRunEvent,
  V2StreamTokenEvent,
  V2ThreadCreateRequest,
  V2ThreadListResponse,
  V2ThreadMessageListResponse,
  V2ThreadRecord,
  V2ThreadUpsertRequest,
  V2RunListResponse,
} from "../types-v2";

export type ApiV2ClientOptions = {
  baseUrl: string;
  apiKey?: string;
};

export class ApiV2Error extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiV2Error";
    this.status = status;
    this.detail = detail;
  }
}

type FetchInit = RequestInit & {
  query?: Record<string, string | number | boolean | null | undefined>;
};

type StreamOptions = V2StreamHandlers & {
  signal?: AbortSignal;
};

const buildUrl = (baseUrl: string, path: string, query?: FetchInit["query"]): string => {
  const url = new URL(path, baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined || value === null) continue;
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
};

async function parseError(response: Response): Promise<never> {
  const text = await response.text();
  let detail: unknown = text;
  try {
    detail = JSON.parse(text);
  } catch {
    // Keep raw text when the backend does not return JSON.
  }
  throw new ApiV2Error(`Request failed with status ${response.status}`, response.status, detail);
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    return parseError(response);
  }
  const text = await response.text();
  if (!text.trim()) {
    return undefined as T;
  }
  return JSON.parse(text) as T;
}

function appendJsonHeaders(init: FetchInit | undefined, apiKey?: string): Headers {
  const headers = new Headers(init?.headers ?? undefined);
  headers.set("Accept", headers.get("Accept") ?? "application/json");
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }
  return headers;
}

function isStreamResponse(response: Response): boolean {
  const contentType = response.headers.get("content-type") ?? "";
  return contentType.includes("text/event-stream");
}

function parseSseData(raw: string): unknown {
  if (!raw.trim()) {
    return {};
  }
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return { message: raw };
  }
}

function buildStreamEvent(eventName: string, dataRaw: string): V2StreamEvent {
  const data = parseSseData(dataRaw);
  const payload = typeof data === "object" && data !== null && !Array.isArray(data)
    ? (data as Record<string, unknown>)
    : {};
  if (eventName === "token") {
    return {
      event: "token",
      data: {
        delta: String(payload.delta ?? ""),
        index: typeof payload.index === "number" ? payload.index : undefined,
        ...payload,
      } as V2StreamTokenEvent["data"],
    };
  }
  if (eventName === "done") {
    return { event: "done", data: payload as V2RunRecord } satisfies V2StreamDoneEvent;
  }
  if (eventName === "error") {
    return {
      event: "error",
      data: {
        message: String(payload.message ?? ""),
        detail: payload.detail as V2StreamErrorEvent["data"]["detail"],
        ...payload,
      } as V2StreamErrorEvent["data"],
    } satisfies V2StreamErrorEvent;
  }
  if (eventName === "run_event") {
    return {
      event: "run_event",
      data: payload as V2GraphEventRecord,
    } satisfies V2StreamRunEvent;
  }
  return { event: eventName, data } satisfies V2StreamEvent;
}

async function* readEventStream(response: Response): AsyncGenerator<V2StreamEvent, void, void> {
  if (!response.body) {
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "message";
  let dataLines: string[] = [];

  const flush = () => {
    if (!dataLines.length && currentEvent === "message") {
      return null;
    }
    const event = buildStreamEvent(currentEvent, dataLines.join("\n"));
    currentEvent = "message";
    dataLines = [];
    return event;
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done }).replace(/\r\n/g, "\n");

    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const line = buffer.slice(0, newlineIndex);
      buffer = buffer.slice(newlineIndex + 1);

      if (!line) {
        const event = flush();
        if (event) {
          yield event;
        }
      } else if (line.startsWith("event:")) {
        currentEvent = line.slice(6).trim() || "message";
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }

      newlineIndex = buffer.indexOf("\n");
    }

    if (done) {
      const tail = flush();
      if (tail) {
        yield tail;
      }
      break;
    }
  }
}

export class ApiV2Client {
  private readonly baseUrl: string;
  private readonly apiKey?: string;

  constructor(options: ApiV2ClientOptions) {
    this.baseUrl = options.baseUrl;
    this.apiKey = options.apiKey?.trim() || undefined;
  }

  private headers(init?: FetchInit): Headers {
    return appendJsonHeaders(init, this.apiKey);
  }

  private async requestJson<T>(path: string, init: FetchInit = {}): Promise<T> {
    const response = await fetch(buildUrl(this.baseUrl, path, init.query), {
      ...init,
      headers: this.headers(init),
      credentials: "include",
    });
    return parseJsonResponse<T>(response);
  }

  private async requestVoid(path: string, init: FetchInit = {}): Promise<void> {
    const response = await fetch(buildUrl(this.baseUrl, path, init.query), {
      ...init,
      headers: this.headers(init),
      credentials: "include",
    });
    if (!response.ok) {
      await parseError(response);
    }
  }

  private async *requestStream<T extends V2StreamEvent>(
    path: string,
    init: FetchInit = {},
    handlers: StreamOptions = {},
  ): AsyncGenerator<T, V2RunRecord | null, void> {
    const headers = this.headers(init);
    headers.set("Accept", "text/event-stream");
    const response = await fetch(buildUrl(this.baseUrl, path, init.query), {
      ...init,
      headers,
      credentials: "include",
      signal: handlers.signal ?? init.signal,
    });

    if (!response.ok) {
      await parseError(response);
    }

    if (!isStreamResponse(response)) {
      const payload = (await response.json()) as V2RunRecord;
      handlers.onDone?.(payload, { event: "done", data: payload });
      handlers.onEvent?.({ event: "done", data: payload });
      return payload;
    }

    let finalRun: V2RunRecord | null = null;
    for await (const event of readEventStream(response)) {
      handlers.onEvent?.(event);
      if (event.event === "token") {
        const tokenEvent = event as V2StreamTokenEvent;
        handlers.onToken?.(tokenEvent.data.delta, tokenEvent);
      } else if (event.event === "run_event") {
        const runEvent = event as V2StreamRunEvent;
        handlers.onRunEvent?.(runEvent.data, runEvent);
      } else if (event.event === "done") {
        const doneEvent = event as V2StreamDoneEvent;
        finalRun = doneEvent.data;
        handlers.onDone?.(doneEvent.data, doneEvent);
      } else if (event.event === "error") {
        const errorEvent = event as V2StreamErrorEvent;
        handlers.onError?.(errorEvent.data, errorEvent);
      }
      yield event as T;
    }
    return finalRun;
  }

  async listThreads(query?: { limit?: number; offset?: number; status?: string }): Promise<V2ThreadListResponse> {
    return this.requestJson<V2ThreadListResponse>("/v2/threads", {
      method: "GET",
      query: {
        limit: query?.limit,
        offset: query?.offset,
        status: query?.status,
      },
    });
  }

  async createThread(request: V2ThreadCreateRequest): Promise<V2ThreadRecord> {
    return this.requestJson<V2ThreadRecord>("/v2/threads", {
      method: "POST",
      body: JSON.stringify(request),
      headers: { "Content-Type": "application/json" },
    });
  }

  async getThread(threadId: string): Promise<V2ThreadRecord> {
    return this.requestJson<V2ThreadRecord>(`/v2/threads/${encodeURIComponent(threadId)}`, {
      method: "GET",
    });
  }

  async upsertThread(threadId: string, request: V2ThreadUpsertRequest): Promise<V2ThreadRecord> {
    return this.requestJson<V2ThreadRecord>(`/v2/threads/${encodeURIComponent(threadId)}`, {
      method: "PUT",
      body: JSON.stringify(request),
      headers: { "Content-Type": "application/json" },
    });
  }

  async listThreadMessages(threadId: string): Promise<V2ThreadMessageListResponse> {
    return this.requestJson<V2ThreadMessageListResponse>(
      `/v2/threads/${encodeURIComponent(threadId)}/messages`,
      { method: "GET" },
    );
  }

  async listRuns(query?: { threadId?: string; status?: string; limit?: number; offset?: number }): Promise<V2RunListResponse> {
    return this.requestJson<V2RunListResponse>("/v2/runs", {
      method: "GET",
      query: {
        thread_id: query?.threadId,
        status: query?.status,
        limit: query?.limit,
        offset: query?.offset,
      },
    });
  }

  async createRun(threadId: string, request: V2RunCreateRequest): Promise<V2RunRecord> {
    return this.requestJson<V2RunRecord>(`/v2/threads/${encodeURIComponent(threadId)}/runs`, {
      method: "POST",
      body: JSON.stringify(request),
      headers: { "Content-Type": "application/json" },
    });
  }

  async streamRun(
    threadId: string,
    request: V2RunCreateRequest,
    handlers: StreamOptions = {},
  ): Promise<V2RunRecord | null> {
    const iterator = this.requestStream<V2StreamEvent>(
      `/v2/threads/${encodeURIComponent(threadId)}/runs`,
      {
        method: "POST",
        body: JSON.stringify(request),
        headers: { "Content-Type": "application/json" },
      },
      handlers,
    );
    let finalRun: V2RunRecord | null = null;
    for await (const event of iterator) {
      if (event.event === "done") {
        finalRun = (event as V2StreamDoneEvent).data;
      }
    }
    return finalRun;
  }

  async getRun(runId: string): Promise<V2RunRecord> {
    return this.requestJson<V2RunRecord>(`/v2/runs/${encodeURIComponent(runId)}`, {
      method: "GET",
    });
  }

  async resumeRun(runId: string, request: V2RunResumeRequest): Promise<V2RunRecord> {
    return this.requestJson<V2RunRecord>(`/v2/runs/${encodeURIComponent(runId)}/resume`, {
      method: "POST",
      body: JSON.stringify(request),
      headers: { "Content-Type": "application/json" },
    });
  }

  async cancelRun(runId: string, request: V2RunCancelRequest = {}): Promise<V2RunRecord> {
    return this.requestJson<V2RunRecord>(`/v2/runs/${encodeURIComponent(runId)}/cancel`, {
      method: "POST",
      body: JSON.stringify(request),
      headers: { "Content-Type": "application/json" },
    });
  }

  async listRunEvents(runId: string, query?: { limit?: number; stream?: boolean }): Promise<V2RunEventsResponse> {
    return this.requestJson<V2RunEventsResponse>(`/v2/runs/${encodeURIComponent(runId)}/events`, {
      method: "GET",
      query: {
        limit: query?.limit,
        stream: query?.stream ? "true" : undefined,
      },
    });
  }

  async streamRunEvents(runId: string, handlers: StreamOptions = {}): Promise<V2RunRecord | null> {
    const iterator = this.requestStream<V2StreamEvent>(
      `/v2/runs/${encodeURIComponent(runId)}/events`,
      {
        method: "GET",
        query: { stream: "true" },
      },
      handlers,
    );
    let finalRun: V2RunRecord | null = null;
    for await (const event of iterator) {
      if (event.event === "done") {
        finalRun = (event as V2StreamDoneEvent).data;
      }
    }
    return finalRun;
  }

  async listRunArtifacts(runId: string, query?: { limit?: number }): Promise<V2ArtifactListResponse> {
    return this.requestJson<V2ArtifactListResponse>(`/v2/runs/${encodeURIComponent(runId)}/artifacts`, {
      method: "GET",
      query: {
        limit: query?.limit,
      },
    });
  }

  async getArtifact(artifactId: string): Promise<V2ArtifactResponse> {
    return this.requestJson<V2ArtifactResponse>(`/v2/artifacts/${encodeURIComponent(artifactId)}`, {
      method: "GET",
    });
  }

  async listArtifactsForRun(runId: string, query?: { limit?: number }): Promise<V2ArtifactRecord[]> {
    const response = await this.listRunArtifacts(runId, query);
    return response.artifacts;
  }

  async deleteThread(threadId: string): Promise<void> {
    await this.requestVoid(`/v2/threads/${encodeURIComponent(threadId)}`, { method: "DELETE" });
  }
}
