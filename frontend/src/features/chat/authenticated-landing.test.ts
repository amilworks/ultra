import { describe, expect, it } from "vitest";

import {
  findAuthenticatedLandingResumeTarget,
  findOrphanComposerDraft,
} from "./authenticated-landing";

const conversation = (
  id: string,
  overrides: Partial<{
    updatedAt: number;
    hydrated: boolean;
    sending: boolean;
    streamingMessageId: string | null;
    historyRunning: boolean;
    prompt: string;
  }> = {}
) => ({
  id,
  updatedAt: overrides.updatedAt ?? 1,
  hydrated: overrides.hydrated ?? true,
  sending: overrides.sending ?? false,
  streamingMessageId: overrides.streamingMessageId ?? null,
  historyRunning: overrides.historyRunning ?? false,
  prompt: overrides.prompt ?? "",
});

describe("authenticated New Chat landing", () => {
  it("prefers the most recent active run over an unsent draft", () => {
    const target = findAuthenticatedLandingResumeTarget(
      [
        conversation("draft", { updatedAt: 30 }),
        conversation("running-old", { updatedAt: 10, historyRunning: true }),
        conversation("running-new", { updatedAt: 20, sending: true }),
      ],
      { draft: "finish this analysis" },
      "landing"
    );

    expect(target).toEqual({ conversationId: "running-new", kind: "active-run" });
  });

  it("offers a non-active conversation with local draft text", () => {
    expect(
      findAuthenticatedLandingResumeTarget(
        [conversation("landing", { updatedAt: 30 }), conversation("draft", { updatedAt: 20 })],
        { landing: "new prompt", draft: "unfinished prompt" },
        "landing"
      )
    ).toEqual({ conversationId: "draft", kind: "draft" });
  });

  it("does not offer whitespace or the active landing conversation as resumable", () => {
    expect(
      findAuthenticatedLandingResumeTarget(
        [conversation("landing", { sending: true }), conversation("empty")],
        { empty: "   " },
        "landing"
      )
    ).toBeNull();
  });

  it("recovers one locally saved draft whose conversation is absent from the server list", () => {
    expect(
      findOrphanComposerDraft(
        {
          server: "server draft",
          stale: "   ",
          local: "unsent local work",
        },
        new Set(["server"])
      )
    ).toEqual({ conversationId: "local", prompt: "unsent local work" });
  });
});
