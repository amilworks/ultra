import type { ApiClient } from "@/lib/api";
import { noteRecoveryScopeFromSession } from "@/lib/noteRecoveryScope";
import type { BisqueAuthSessionResponse } from "@/types";

export type AuthenticatedAccountDeparture = "logout" | "unlink";

export type AuthenticatedBisqueUnlinkResult =
  | {
      kind: "session-preserved";
      recoveryScope: string;
      session: BisqueAuthSessionResponse;
    }
  | {
      kind: "principal-changed";
      recoveryScope: string;
      session: BisqueAuthSessionResponse;
    }
  | { kind: "signed-out"; session: BisqueAuthSessionResponse }
  | { kind: "indeterminate" };

const classifyAuthenticatedBisqueUnlinkSession = (
  session: BisqueAuthSessionResponse,
  currentRecoveryScope: string | null
): Exclude<AuthenticatedBisqueUnlinkResult, { kind: "indeterminate" }> | null => {
  if (session.authenticated === false) {
    return { kind: "signed-out", session };
  }
  if (session.authenticated !== true || session.bisque_linked !== false) {
    return null;
  }
  if (session.provider !== "workos" || session.mode !== "workos") {
    return null;
  }
  const recoveryScope = noteRecoveryScopeFromSession(session);
  if (!recoveryScope) {
    return null;
  }
  return recoveryScope === currentRecoveryScope
    ? { kind: "session-preserved", recoveryScope, session }
    : { kind: "principal-changed", recoveryScope, session };
};

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

/**
 * Unlinking BisQue from a WorkOS account is not an account departure. Preserve
 * the authoritative session response so the caller can remove only the linked
 * credential state. Local BisQue auth may still become signed out.
 */
export const requestAuthenticatedBisqueUnlink = async (
  apiClient: Pick<ApiClient, "getBisqueSession" | "unlinkBisqueAccount">,
  currentRecoveryScope: string | null
): Promise<AuthenticatedBisqueUnlinkResult> => {
  const session = await apiClient.unlinkBisqueAccount();
  const result = classifyAuthenticatedBisqueUnlinkSession(session, currentRecoveryScope);
  if (result) {
    return result;
  }

  // The unlink response is authoritative in current servers, but the API
  // schema allows older deployments to omit its link projection. Reconcile
  // once through the session endpoint before failing closed.
  try {
    const reconciledSession = await apiClient.getBisqueSession();
    return (
      classifyAuthenticatedBisqueUnlinkSession(
        reconciledSession,
        currentRecoveryScope
      ) ?? { kind: "indeterminate" }
    );
  } catch {
    return { kind: "indeterminate" };
  }
};

/** Keep every async unlink boundary bound to the account that initiated it. */
export const performAuthenticatedBisqueUnlink = async <
  PreparedState extends { recoveryScope: string | null },
>({
  apiClient,
  ownerScope,
  currentOwnerScope,
  prepareOwnerState,
  applyResult,
}: {
  apiClient: Pick<ApiClient, "getBisqueSession" | "unlinkBisqueAccount">;
  ownerScope: string | null;
  currentOwnerScope: () => string | null;
  prepareOwnerState: () => Promise<PreparedState | null>;
  applyResult: (
    result: AuthenticatedBisqueUnlinkResult,
    prepared: PreparedState
  ) => void;
}): Promise<void> => {
  const stillOwnsTransition = () => currentOwnerScope() === ownerScope;
  if (!stillOwnsTransition()) return;
  let prepared: PreparedState | null;
  let result: AuthenticatedBisqueUnlinkResult;
  try {
    prepared = await prepareOwnerState();
    if (
      !prepared ||
      prepared.recoveryScope !== ownerScope ||
      !stillOwnsTransition()
    ) {
      return;
    }
    result = await requestAuthenticatedBisqueUnlink(apiClient, ownerScope);
  } catch (error) {
    // The caller's error UI belongs to the initiating account too.
    if (stillOwnsTransition()) throw error;
    return;
  }
  if (stillOwnsTransition()) {
    applyResult(result, prepared);
  }
};
