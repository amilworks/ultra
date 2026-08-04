/**
 * Run event stream reconnection (consumeV2RunEventStream orchestration).
 *
 * A run outlives any single SSE connection: laptops sleep overnight, proxies and
 * the control plane restart, networks blip. These tests script fetch to serve
 * SSE bodies that die mid-run and assert the client resumes at the exact event
 * cursor, never duplicates a token, settles from the run record when the run
 * finished while disconnected, and still fails fast on non-retryable answers.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClient, ApiError } from "./api";

const BASE = "https://ultra.example.org";
const RUN_ID = "run_reconnect_test";

const sse = (...blocks: string[]): string => blocks.map((block) => `${block}\n\n`).join("");

const runEventBlock = (sequence: number, eventKind: string, extra: Record<string, unknown> = {}): string =>
  `event: run_event\ndata: ${JSON.stringify({ sequence, event_kind: eventKind, run_id: RUN_ID, ...extra })}`;

const tokenBlock = (sequence: number, delta: string): string =>
  runEventBlock(sequence, "message.delta", { payload: { delta } });

// A ReadableStream that yields the given text then either closes cleanly, errors,
// or hangs. Pull-based so the text chunk is guaranteed to be DELIVERED before the
// error/hang — controller.error() in start() would poison the queued chunk too.
const streamBody = (
  text: string,
  end: "close" | "error" | "hang" = "close"
): ReadableStream<Uint8Array> => {
  const encoder = new TextEncoder();
  let step = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (step === 0) {
        step = 1;
        if (text) {
          controller.enqueue(encoder.encode(text));
          return;
        }
      }
      if (step === 1) {
        step = 2;
        if (end === "close") {
          controller.close();
        } else if (end === "error") {
          controller.error(new TypeError("network connection was interrupted"));
        }
        // "hang": leave the pull pending forever — the dead-but-open socket.
      }
    },
  });
};

const sseResponse = (body: ReadableStream<Uint8Array>): Response =>
  new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });

const jsonResponse = (payload: unknown, status = 200): Response =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });

type FetchPlan = Array<(url: string) => Response | Promise<Response>>;

const installFetchPlan = (plan: FetchPlan): string[] => {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      calls.push(url);
      const step = plan.shift();
      if (!step) {
        throw new Error(`fetch plan exhausted for ${url}`);
      }
      return step(url);
    })
  );
  return calls;
};

const fastOptions = { retryBaseDelayMs: 1, inactivityTimeoutMs: 1200 } as const;

describe("consumeV2RunEventStream reconnection", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("resumes at the cursor after a mid-run cut and never duplicates tokens", async () => {
    const client = new ApiClient({ baseUrl: BASE });
    const tokens: string[] = [];

    const calls = installFetchPlan([
      // Attempt 1: two tokens, then the connection errors (proxy died mid-body).
      // A thrown attempt retries the stream DIRECTLY (no status probe: the
      // error told us nothing about the run, and the reconnect is the probe).
      () => sseResponse(streamBody(sse(tokenBlock(1, "Hello "), tokenBlock(2, "wor")), "error")),
      // Attempt 2 resumes; the server replays an overlap (sequence 2)
      // which the cross-attempt dedupe must drop, then finishes the run.
      () =>
        sseResponse(
          streamBody(
            sse(
              tokenBlock(2, "wor"),
              tokenBlock(3, "ld."),
              runEventBlock(4, "run.completed", { payload: { response_text: "Hello world." } })
            )
          )
        ),
      // Terminal settle: authoritative run record.
      () => jsonResponse({ run_id: RUN_ID, status: "succeeded", response_text: "Hello world." }),
    ]);

    const result = await client.resumeRunStream(RUN_ID, {
      ...fastOptions,
      onToken: (delta) => tokens.push(delta),
    });

    expect(tokens.join("")).toBe("Hello world.");
    expect(result.response_text).toBe("Hello world.");
    const streamCalls = calls.filter((url) => url.includes("stream=true"));
    expect(streamCalls).toHaveLength(2);
    expect(streamCalls[0]).toContain("after_sequence=0");
    expect(streamCalls[1]).toContain("after_sequence=2");
  });

  it("settles from the run record when the run finished while disconnected", async () => {
    const client = new ApiClient({ baseUrl: BASE });

    const calls = installFetchPlan([
      // Attempt 1: a token, then a clean close with no terminal event (the
      // wake-after-overnight shape: the original stream is long gone).
      () => sseResponse(streamBody(sse(tokenBlock(1, "partial")))),
      // Status probe: the run reached terminal while we were away.
      () =>
        jsonResponse({
          run_id: RUN_ID,
          status: "succeeded",
          response_text: "Finished overnight.",
        }),
    ]);

    const result = await client.resumeRunStream(RUN_ID, { ...fastOptions });

    expect(result.response_text).toBe("Finished overnight.");
    // No second stream attempt: terminal settles directly.
    expect(calls.filter((url) => url.includes("stream=true"))).toHaveLength(1);
  });

  it("keeps retrying while the run stays running across clean severs", async () => {
    const client = new ApiClient({ baseUrl: BASE });

    installFetchPlan([
      () => sseResponse(streamBody("")), // instant sever, nothing delivered
      () => jsonResponse({ run_id: RUN_ID, status: "running" }),
      () => sseResponse(streamBody("")), // sever again
      () => jsonResponse({ run_id: RUN_ID, status: "running" }),
      () =>
        sseResponse(
          streamBody(sse(runEventBlock(9, "run.completed", { payload: { response_text: "done" } })))
        ),
      () => jsonResponse({ run_id: RUN_ID, status: "succeeded", response_text: "done" }),
    ]);

    const result = await client.resumeRunStream(RUN_ID, { ...fastOptions });
    expect(result.response_text).toBe("done");
  });

  it("cancels a silent connection via the inactivity watchdog and reconnects", async () => {
    const client = new ApiClient({ baseUrl: BASE });

    const calls = installFetchPlan([
      // Attempt 1 delivers one token then goes silent forever (half-open TCP
      // after OS sleep). The watchdog must cancel it — nothing else will.
      () => sseResponse(streamBody(sse(tokenBlock(1, "before-sleep ")), "hang")),
      () => jsonResponse({ run_id: RUN_ID, status: "running" }),
      () =>
        sseResponse(
          streamBody(
            sse(
              tokenBlock(2, "after-wake"),
              runEventBlock(3, "run.completed", { payload: { response_text: "before-sleep after-wake" } })
            )
          )
        ),
      () => jsonResponse({ run_id: RUN_ID, status: "succeeded" }),
    ]);

    const tokens: string[] = [];
    const result = await client.resumeRunStream(RUN_ID, {
      retryBaseDelayMs: 1,
      inactivityTimeoutMs: 150,
      onToken: (delta) => tokens.push(delta),
    });

    expect(tokens.join("")).toBe("before-sleep after-wake");
    expect(result.response_text).toBe("before-sleep after-wake");
    expect(calls.filter((url) => url.includes("stream=true"))).toHaveLength(2);
  }, 15_000);

  it("throws immediately on non-retryable answers (auth) without reconnecting", async () => {
    const client = new ApiClient({ baseUrl: BASE });

    const calls = installFetchPlan([() => jsonResponse({ detail: "who are you" }, 401)]);

    await expect(client.resumeRunStream(RUN_ID, { ...fastOptions })).rejects.toMatchObject({
      status: 401,
    });
    expect(calls).toHaveLength(1);
  });

  it("propagates the caller's abort instead of retrying", async () => {
    const client = new ApiClient({ baseUrl: BASE });
    const controller = new AbortController();

    installFetchPlan([
      () => {
        controller.abort();
        return Promise.reject(new DOMException("The operation was aborted.", "AbortError"));
      },
    ]);

    await expect(
      client.resumeRunStream(RUN_ID, { ...fastOptions, signal: controller.signal })
    ).rejects.toMatchObject({ name: "AbortError" });
  });

  it("still reports a genuinely failed run as failed", async () => {
    const client = new ApiClient({ baseUrl: BASE });

    installFetchPlan([
      () =>
        sseResponse(
          streamBody(sse(runEventBlock(5, "run.failed", { payload: { error: "sandbox exploded" } })))
        ),
    ]);

    await expect(client.resumeRunStream(RUN_ID, { ...fastOptions })).rejects.toMatchObject({
      message: "Run failed",
      status: 500,
    });
  });
});

// The ApiError import participates in type-level assertions above.
void ApiError;
