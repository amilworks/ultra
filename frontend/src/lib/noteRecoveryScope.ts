import type { BisqueAuthSessionResponse } from "@/types";

const exactPart = (value: unknown): string => String(value ?? "").trim();

/**
 * A storage namespace derived from the authenticated Ultra identity, never
 * from mutable display text such as a newly linked BisQue username.
 */
export const noteRecoveryScopeFromSession = (
  session: BisqueAuthSessionResponse
): string | null => {
  if (!session.authenticated) return null;

  const accountId = exactPart(session.account_user_id);
  if (accountId) return `v1:account:${accountId}`;

  const principalId = exactPart(session.user?.id);
  if (!principalId) return null;
  const provider =
    session.provider === "workos" || session.mode === "workos" ? "workos" : "local";
  return `v1:principal:${provider}:${principalId}`;
};
