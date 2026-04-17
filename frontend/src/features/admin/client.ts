import type { ApiClient } from "@/lib/api";
import type {
  AdminIssueListResponse,
  AdminOverviewResponse,
  AdminRunListResponse,
  AdminUserListResponse,
} from "@/types";

export type AdminOverviewQuery = {
  topUsers?: number;
  issueLimit?: number;
};

export type AdminRunQuery = {
  limit?: number;
  offset?: number;
  query?: string;
  status?: string;
  userId?: string;
};

export type AdminUserQuery = {
  limit?: number;
  query?: string;
};

export const loadAdminOverview = (
  apiClient: ApiClient,
  query: AdminOverviewQuery = {}
): Promise<AdminOverviewResponse> =>
  apiClient.getAdminOverview({
    topUsers: query.topUsers ?? 8,
    issueLimit: query.issueLimit ?? 12,
  });

export const loadAdminUsers = (
  apiClient: ApiClient,
  query: AdminUserQuery = {}
): Promise<AdminUserListResponse> =>
  apiClient.listAdminUsers({
    limit: query.limit ?? 250,
    query: query.query,
  });

export const loadAdminRuns = (
  apiClient: ApiClient,
  query: AdminRunQuery = {}
): Promise<AdminRunListResponse> =>
  apiClient.listAdminRuns({
    limit: query.limit ?? 100,
    offset: query.offset ?? 0,
    query: query.query,
    status: query.status,
    userId: query.userId,
  });

export const loadAdminIssues = (
  apiClient: ApiClient,
  limit = 25
): Promise<AdminIssueListResponse> => apiClient.listAdminIssues(limit);
