import type { ApiClient } from "@/lib/api";
import type {
  PrairieRetrainRecord,
  PrairieStatusResponse,
  TrainingLineageRecord,
  TrainingModelRecord,
  TrainingModelVersionRecord,
} from "@/types";

export type TrainingLineageSnapshot = {
  lineage: TrainingLineageRecord | null;
  versions: TrainingModelVersionRecord[];
};

export type TrainingDashboardSnapshot = {
  models: TrainingModelRecord[];
  status: PrairieStatusResponse;
  retrainRequests: PrairieRetrainRecord[];
};

export const loadTrainingDashboardSnapshot = async (
  apiClient: ApiClient
): Promise<TrainingDashboardSnapshot> => {
  const [modelsPayload, status, retrainPayload] = await Promise.all([
    apiClient.listTrainingModels(),
    apiClient.getPrairieActiveLearningStatus(),
    apiClient.listPrairieRetrainRequests(),
  ]);
  return {
    models: modelsPayload.models,
    retrainRequests: retrainPayload.requests,
    status,
  };
};

export const loadTrainingLineageSnapshot = async (
  apiClient: ApiClient,
  modelKey: string,
  options?: {
    domainLimit?: number;
    lineageLimit?: number;
    versionLimit?: number;
  }
): Promise<TrainingLineageSnapshot> => {
  const domains = await apiClient.listTrainingDomains(options?.domainLimit ?? 500);
  for (const domain of domains.domains) {
    const lineages = await apiClient.listDomainLineages(domain.domain_id, {
      limit: options?.lineageLimit ?? 500,
    });
    const found =
      lineages.lineages.find(
        (row) => row.model_key === modelKey && row.scope === "shared"
      ) ?? lineages.lineages.find((row) => row.model_key === modelKey) ?? null;
    if (!found) {
      continue;
    }
    const versionsPayload = await apiClient.listLineageVersions(found.lineage_id, {
      limit: options?.versionLimit ?? 200,
    });
    return {
      lineage: found,
      versions: versionsPayload.versions,
    };
  }
  return {
    lineage: null,
    versions: [],
  };
};
