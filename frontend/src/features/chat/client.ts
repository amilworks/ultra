import type { ApiClient } from "@/lib/api";
import type {
  ArtifactListResponse,
  ConversationListResponse,
  RunEventsResponse,
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
  limit = 200,
  options?: { afterSequence?: number }
): Promise<RunEventsResponse> => apiClient.getRunEvents(runId, limit, options);

export const listRunArtifacts = (
  apiClient: ApiClient,
  runId: string,
  limit = 500
): Promise<ArtifactListResponse> => apiClient.listArtifacts(runId, limit);
