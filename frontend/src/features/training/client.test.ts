import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import {
  loadTrainingDashboardSnapshot,
  loadTrainingLineageSnapshot,
} from "./client";

describe("training slice client", () => {
  it("loads the prairie dashboard snapshot in one call", async () => {
    const apiClient = {
      listTrainingModels: vi.fn().mockResolvedValue({ models: [{ key: "a", supports_inference: true }] }),
      getPrairieActiveLearningStatus: vi.fn().mockResolvedValue({ benchmark_ready: true }),
      listPrairieRetrainRequests: vi.fn().mockResolvedValue({ requests: [{ request_id: "r1" }] }),
    } as unknown as ApiClient;

    const snapshot = await loadTrainingDashboardSnapshot(apiClient);

    expect(snapshot.models).toHaveLength(1);
    expect(snapshot.retrainRequests).toHaveLength(1);
    expect(apiClient.listTrainingModels).toHaveBeenCalledTimes(1);
  });

  it("finds the shared lineage for a model and returns its versions", async () => {
    const apiClient = {
      listTrainingDomains: vi.fn().mockResolvedValue({
        domains: [{ domain_id: "domain-1" }],
      }),
      listDomainLineages: vi.fn().mockResolvedValue({
        lineages: [
          { lineage_id: "lineage-1", model_key: "prairie", scope: "shared" },
        ],
      }),
      listLineageVersions: vi.fn().mockResolvedValue({
        versions: [{ version_id: "version-1" }],
      }),
    } as unknown as ApiClient;

    const snapshot = await loadTrainingLineageSnapshot(apiClient, "prairie");

    expect(snapshot.lineage?.lineage_id).toBe("lineage-1");
    expect(snapshot.versions).toEqual([{ version_id: "version-1" }]);
    expect(apiClient.listLineageVersions).toHaveBeenCalledWith("lineage-1", {
      limit: 200,
    });
  });
});
