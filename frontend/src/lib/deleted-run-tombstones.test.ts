/**
 * A deleted turn must stay deleted across a reload.
 *
 * Hydration reconciles a saved snapshot against the durable transcript, and its
 * staleness test is "is there an assistant message carrying
 * thread.latest_run_id?". Deleting the most recent turn creates exactly that
 * condition, so reconciliation concluded the snapshot was stale and rebuilt the
 * deleted answer from control_runs.response_text — the message came back on the
 * next load and the delete looked broken.
 *
 * Worse, findAssistantPatchIndex falls back to "last assistant with no runId",
 * so the resurrected answer could be written over an unrelated surviving
 * message rather than merely reappearing.
 *
 * `deletedRunIds` in the persisted state is the tombstone that stops both.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiClient } from "./api";

const BASE = "https://ultra.example.org";

type FrontendState = Record<string, unknown>;

const threadResponse = (state: FrontendState) =>
  new Response(
    JSON.stringify({
      thread_id: "thread_del",
      title: "sensitive analysis",
      status: "active",
      created_at: "2026-07-25T10:00:00Z",
      updated_at: "2026-07-25T10:05:00Z",
      latest_run_id: "run_deleted",
      metadata: { conversation_id: "conversation-del", frontend_state: state },
    }),
    { status: 200, headers: { "Content-Type": "application/json" } }
  );

// The durable side still knows the answer — that is the whole point. Hard delete
// removes it eventually, but a message deleted from a surviving conversation
// lives on in control_runs until the conversation itself goes.
const durableMessages = () =>
  new Response(
    JSON.stringify({
      thread_id: "thread_del",
      count: 1,
      messages: [
        {
          message_id: "msg_durable_user",
          role: "user",
          content: "something private",
          created_at: "2026-07-25T10:00:00Z",
          run_id: "run_deleted",
        },
      ],
    }),
    { status: 200, headers: { "Content-Type": "application/json" } }
  );

const runResponse = () =>
  new Response(
    JSON.stringify({
      run_id: "run_deleted",
      thread_id: "thread_del",
      status: "succeeded",
      response_text: "THE DELETED ANSWER",
      updated_at: "2026-07-25T10:05:00Z",
      metadata: {},
    }),
    { status: 200, headers: { "Content-Type": "application/json" } }
  );

const mockFetch = (state: FrontendState) => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === `${BASE}/v2/threads/thread_del`) return threadResponse(state);
    if (url === `${BASE}/v2/threads/thread_del/messages`) return durableMessages();
    if (url === `${BASE}/v2/runs/run_deleted`) return runResponse();
    throw new Error(`Unexpected fetch: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

const seedThreadMap = () => {
  window.localStorage.setItem(
    "bisque-ultra:v2-chat-thread-map",
    JSON.stringify({ "conversation-del": "thread_del" })
  );
};

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

describe("deleted run tombstones", () => {
  it("resurrects a deleted answer when there is no tombstone (the bug)", async () => {
    // Pinning the broken behaviour deliberately: if this ever stops
    // resurrecting, the tombstone test below is no longer proving anything.
    seedThreadMap();
    mockFetch({
      messages: [{ id: "msg-keep", role: "user", content: "unrelated later question" }],
    });

    const client = new ApiClient({ baseUrl: BASE });
    const response = await client.getConversation("conversation-del");

    expect(JSON.stringify(response.state)).toContain("THE DELETED ANSWER");
  });

  it("keeps a tombstoned run deleted across a reload", async () => {
    seedThreadMap();
    mockFetch({
      messages: [{ id: "msg-keep", role: "user", content: "unrelated later question" }],
      deletedRunIds: ["run_deleted"],
    });

    const client = new ApiClient({ baseUrl: BASE });
    const response = await client.getConversation("conversation-del");

    expect(JSON.stringify(response.state)).not.toContain("THE DELETED ANSWER");
  });

  it("does not overwrite an unrelated surviving message with the deleted answer", async () => {
    // findAssistantPatchIndex falls back to "last assistant with no runId", so
    // an untombstoned resurrection can land on the wrong message entirely.
    seedThreadMap();
    mockFetch({
      messages: [
        { id: "msg-keep-user", role: "user", content: "unrelated later question" },
        { id: "msg-keep-assistant", role: "assistant", content: "AN UNRELATED SURVIVING ANSWER" },
      ],
      deletedRunIds: ["run_deleted"],
    });

    const client = new ApiClient({ baseUrl: BASE });
    const response = await client.getConversation("conversation-del");

    const serialized = JSON.stringify(response.state);
    expect(serialized).toContain("AN UNRELATED SURVIVING ANSWER");
    expect(serialized).not.toContain("THE DELETED ANSWER");
  });

  it("ignores malformed tombstone data rather than throwing during hydration", async () => {
    // This value round-trips through thread metadata, so it must be treated as
    // untrusted input — a hydration crash would lock the user out of the chat.
    seedThreadMap();
    mockFetch({
      messages: [{ id: "msg-keep", role: "user", content: "hello" }],
      deletedRunIds: "not-an-array",
    });

    const client = new ApiClient({ baseUrl: BASE });
    await expect(client.getConversation("conversation-del")).resolves.toBeTruthy();
  });
});
