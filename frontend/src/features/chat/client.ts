import type { ApiClient } from "@/lib/api";
import type {
  ArtifactListResponse,
  ConversationListResponse,
  RunEventsResponse,
  V3RunCreateRequest,
  V3RunRecord,
  V3SessionCreateRequest,
  V3SessionRecord,
} from "@/types";

export type ConversationPageQuery = {
  limit?: number;
  offset?: number;
  includeState?: boolean;
};

export const listSessionConversations = (
  apiClient: ApiClient,
  query: ConversationPageQuery = {}
): Promise<ConversationListResponse> =>
  apiClient.listConversations(
    query.limit ?? 25,
    query.offset ?? 0,
    query.includeState ?? false
  );

export const listRunEvents = (
  apiClient: ApiClient,
  runId: string,
  limit = 200
): Promise<RunEventsResponse> => apiClient.getRunEvents(runId, limit);

export const listRunArtifacts = (
  apiClient: ApiClient,
  runId: string,
  limit = 500
): Promise<ArtifactListResponse> => apiClient.listArtifacts(runId, limit);

export const createScientistSession = (
  apiClient: ApiClient,
  request: V3SessionCreateRequest
): Promise<V3SessionRecord> => apiClient.createV3Session(request);

export const createScientistRun = (
  apiClient: ApiClient,
  sessionId: string,
  request: V3RunCreateRequest
): Promise<V3RunRecord> => apiClient.createV3Run(sessionId, request);
