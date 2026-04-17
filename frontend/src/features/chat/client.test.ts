import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import {
  createScientistRun,
  createScientistSession,
  listRunArtifacts,
  listRunEvents,
  listSessionConversations,
} from "./client";

describe("chat slice client", () => {
  it("shapes conversation paging defaults in one place", async () => {
    const apiClient = {
      listConversations: vi.fn().mockResolvedValue({
        count: 0,
        conversations: [],
        has_more: false,
        offset: 0,
      }),
    } as unknown as ApiClient;

    await listSessionConversations(apiClient, { limit: 50, offset: 10 });

    expect(apiClient.listConversations).toHaveBeenCalledWith(50, 10, false);
  });

  it("delegates run hydration and v3 calls through the chat slice", async () => {
    const apiClient = {
      getRunEvents: vi.fn().mockResolvedValue({ count: 0, events: [], run_id: "run-1" }),
      listArtifacts: vi.fn().mockResolvedValue({ artifacts: [], count: 0, run_id: "run-1" }),
      createV3Session: vi.fn().mockResolvedValue({ session_id: "session-1" }),
      createV3Run: vi.fn().mockResolvedValue({ run_id: "run-1" }),
    } as unknown as ApiClient;

    await listRunEvents(apiClient, "run-1", 120);
    await listRunArtifacts(apiClient, "run-1", 2000);
    await createScientistSession(apiClient, {});
    await createScientistRun(apiClient, "session-1", { messages: [] });

    expect(apiClient.getRunEvents).toHaveBeenCalledWith("run-1", 120);
    expect(apiClient.listArtifacts).toHaveBeenCalledWith("run-1", 2000);
    expect(apiClient.createV3Session).toHaveBeenCalledWith({});
    expect(apiClient.createV3Run).toHaveBeenCalledWith("session-1", { messages: [] });
  });
});
