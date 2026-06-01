import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import {
  createAdminOrganization,
  createAdminUser,
  deleteAdminUser,
  loadAdminOrganizations,
  loadAdminIssues,
  loadAdminOverview,
  loadAdminRuns,
  loadAdminUsers,
  requeueAdminRun,
} from "./client";

describe("admin slice client", () => {
  it("keeps admin defaults centralized", async () => {
    const apiClient = {
      getAdminOverview: vi.fn().mockResolvedValue({}),
      createAdminOrganization: vi.fn().mockResolvedValue({ org_id: "allen-institute" }),
      listAdminOrganizations: vi.fn().mockResolvedValue({ count: 0, organizations: [] }),
      createAdminUser: vi.fn().mockResolvedValue({ user_id: "user_1" }),
      deleteAdminUser: vi.fn().mockResolvedValue({ user_id: "user_1", status: "disabled" }),
      listAdminUsers: vi.fn().mockResolvedValue({ count: 0, users: [] }),
      listAdminRuns: vi.fn().mockResolvedValue({ count: 0, runs: [] }),
      listAdminIssues: vi.fn().mockResolvedValue({ count: 0, issues: [] }),
      requeueAdminRun: vi.fn().mockResolvedValue({ run_id: "run_1", updated: true }),
    } as unknown as ApiClient;

    await loadAdminOverview(apiClient);
    await loadAdminOrganizations(apiClient, { query: "allen" });
    await createAdminOrganization(apiClient, {
      org_id: "allen-institute",
      name: "Allen Institute",
      status: "active",
    });
    await loadAdminUsers(apiClient, { query: "researcher" });
    await createAdminUser(apiClient, {
      email: "ada@example.org",
      display_name: "Ada Lovelace",
      role: "admin",
      org_id: "local-org",
    });
    await deleteAdminUser(apiClient, "user_1");
    await loadAdminRuns(apiClient, { status: "running" });
    await loadAdminIssues(apiClient);
    await requeueAdminRun(apiClient, "run_1", "expired lease");

    expect(apiClient.getAdminOverview).toHaveBeenCalledWith({
      issueLimit: 12,
      topUsers: 8,
    });
    expect(apiClient.listAdminOrganizations).toHaveBeenCalledWith({
      limit: 250,
      query: "allen",
    });
    expect(apiClient.createAdminOrganization).toHaveBeenCalledWith({
      org_id: "allen-institute",
      name: "Allen Institute",
      status: "active",
    });
    expect(apiClient.listAdminUsers).toHaveBeenCalledWith({
      limit: 250,
      query: "researcher",
    });
    expect(apiClient.createAdminUser).toHaveBeenCalledWith({
      email: "ada@example.org",
      display_name: "Ada Lovelace",
      role: "admin",
      org_id: "local-org",
    });
    expect(apiClient.deleteAdminUser).toHaveBeenCalledWith("user_1");
    expect(apiClient.listAdminRuns).toHaveBeenCalledWith({
      limit: 100,
      offset: 0,
      query: undefined,
      status: "running",
      userId: undefined,
    });
    expect(apiClient.listAdminIssues).toHaveBeenCalledWith(25);
    expect(apiClient.requeueAdminRun).toHaveBeenCalledWith("run_1", "expired lease");
  });
});
