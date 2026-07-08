import type { ApiClient } from "@/lib/api";
import type {
  PrairieRetrainRecord,
  PrairieStatusResponse,
  TrainingLineageRecord,
  TrainingModelRecord,
  TrainingModelStatus,
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

// --- GoldGate answer page (M1.5) --------------------------------------------

export type TrainingModelSnapshot = {
  models: TrainingModelRecord[];
  model: TrainingModelRecord | null;
  status: TrainingModelStatus;
  lineage: TrainingLineageRecord | null;
  versions: TrainingModelVersionRecord[];
  retrainRequests: PrairieRetrainRecord[];
};

// The ONE prairie-shaped read hides behind this seam: the M1.5 PR 3 cutover
// swaps its body to the generic GET /v2/training/models/{key}/status and
// nothing else changes (plan section 14.10). Every echo field is optional -
// deriveTrainingPhase degrades per the section-14.5 table when absent.
export const getModelStatus = async (
  apiClient: ApiClient,
  modelKey: string
): Promise<TrainingModelStatus> => {
  void modelKey; // routes to the prairie alias until the PR 3 cutover
  const status = await apiClient.getPrairieActiveLearningStatus();
  return status as TrainingModelStatus;
};

// Model-scoped snapshot for the rebuilt dashboard. Fetch fan-out and default
// limits are IDENTICAL to the legacy loaders (client.test.ts pins them).
export const loadTrainingSnapshot = async (
  apiClient: ApiClient,
  modelKey: string,
  options?: {
    domainLimit?: number;
    lineageLimit?: number;
    versionLimit?: number;
  }
): Promise<TrainingModelSnapshot> => {
  const [modelsPayload, status, retrainPayload, lineageSnapshot] = await Promise.all([
    apiClient.listTrainingModels(),
    getModelStatus(apiClient, modelKey),
    apiClient.listPrairieRetrainRequests(),
    loadTrainingLineageSnapshot(apiClient, modelKey, options),
  ]);
  return {
    models: modelsPayload.models,
    model: modelsPayload.models.find((row) => row.key === modelKey) ?? null,
    status,
    lineage: lineageSnapshot.lineage,
    versions: lineageSnapshot.versions,
    retrainRequests: retrainPayload.requests,
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
