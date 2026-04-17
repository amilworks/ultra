import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import {
  loadAdminIssues,
  loadAdminOverview,
  loadAdminRuns,
  loadAdminUsers,
} from "./client";

describe("admin slice client", () => {
  it("keeps admin defaults centralized", async () => {
    const apiClient = {
      getAdminOverview: vi.fn().mockResolvedValue({}),
      listAdminUsers: vi.fn().mockResolvedValue({ count: 0, users: [] }),
      listAdminRuns: vi.fn().mockResolvedValue({ count: 0, runs: [] }),
      listAdminIssues: vi.fn().mockResolvedValue({ count: 0, issues: [] }),
    } as unknown as ApiClient;

    await loadAdminOverview(apiClient);
    await loadAdminUsers(apiClient, { query: "researcher" });
    await loadAdminRuns(apiClient, { status: "running" });
    await loadAdminIssues(apiClient);

    expect(apiClient.getAdminOverview).toHaveBeenCalledWith({
      issueLimit: 12,
      topUsers: 8,
    });
    expect(apiClient.listAdminUsers).toHaveBeenCalledWith({
      limit: 250,
      query: "researcher",
    });
    expect(apiClient.listAdminRuns).toHaveBeenCalledWith({
      limit: 100,
      offset: 0,
      query: undefined,
      status: "running",
      userId: undefined,
    });
    expect(apiClient.listAdminIssues).toHaveBeenCalledWith(25);
  });
});
