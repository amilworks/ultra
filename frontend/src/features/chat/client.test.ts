import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import {
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

  it("delegates V2 run hydration through the chat slice", async () => {
    const apiClient = {
      getRunEvents: vi.fn().mockResolvedValue({ count: 0, events: [], run_id: "run-1" }),
      listArtifacts: vi.fn().mockResolvedValue({ artifacts: [], count: 0, run_id: "run-1" }),
    } as unknown as ApiClient;

    await listRunEvents(apiClient, "run-1", 120);
    await listRunEvents(apiClient, "run-1", 120, { afterSequence: 40 });
    await listRunArtifacts(apiClient, "run-1", 2000);

    expect(apiClient.getRunEvents).toHaveBeenNthCalledWith(1, "run-1", 120, undefined);
    expect(apiClient.getRunEvents).toHaveBeenNthCalledWith(2, "run-1", 120, { afterSequence: 40 });
    expect(apiClient.listArtifacts).toHaveBeenCalledWith("run-1", 2000);
  });
});
