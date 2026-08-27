import type { ApiClient } from "@/lib/api";

export type AuthenticatedAccountDeparture = "logout" | "unlink";

/**
 * Requests the authoritative server-side account transition. Rejections are
 * deliberately not swallowed: the caller must keep the authenticated UI and
 * owner-bound state until the server confirms departure.
 */
export const requestAuthenticatedAccountDeparture = async (
  apiClient: Pick<ApiClient, "logoutBisque" | "unlinkBisqueAccount">,
  departure: AuthenticatedAccountDeparture
): Promise<string> => {
  if (departure === "logout") {
    const session = await apiClient.logoutBisque();
    return String(session.logout_url ?? "").trim();
  }
  await apiClient.unlinkBisqueAccount();
  return "";
};
