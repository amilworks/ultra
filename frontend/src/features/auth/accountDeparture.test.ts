import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import { noteRecoveryScopeFromSession } from "@/lib/noteRecoveryScope";
import type { BisqueAuthSessionResponse } from "@/types";
import {
  performAuthenticatedBisqueUnlink,
  requestAuthenticatedAccountDeparture,
  requestAuthenticatedBisqueUnlink,
} from "./accountDeparture";

describe("authenticated account departure", () => {
  it.each(["logout", "unlink"] as const)(
    "does not turn a failed %s request into local success",
    async (departure) => {
      const failure = new Error(`${departure} unavailable`);
      const apiClient = {
        logoutBisque: vi.fn().mockRejectedValue(failure),
        unlinkBisqueAccount: vi.fn().mockRejectedValue(failure),
      } as unknown as ApiClient;

      await expect(
        requestAuthenticatedAccountDeparture(apiClient, departure)
      ).rejects.toBe(failure);
      expect(apiClient.logoutBisque).toHaveBeenCalledTimes(departure === "logout" ? 1 : 0);
      expect(apiClient.unlinkBisqueAccount).toHaveBeenCalledTimes(
        departure === "unlink" ? 1 : 0
      );
    }
  );

  it("returns a logout redirect only after the server confirms it", async () => {
    const apiClient = {
      logoutBisque: vi.fn().mockResolvedValue({ logout_url: " /signed-out " }),
      unlinkBisqueAccount: vi.fn(),
    } as unknown as ApiClient;

    await expect(
      requestAuthenticatedAccountDeparture(apiClient, "logout")
    ).resolves.toBe("/signed-out");
  });

  it.each([
    {
      expectedKind: "session-preserved",
      label: "the same WorkOS principal",
      session: {
        authenticated: true,
        provider: "workos",
        mode: "workos",
        account_user_id: "account-a",
        user: { id: "user-a", email: "researcher@example.org" },
        bisque_linked: false,
      },
    },
    {
      expectedKind: "signed-out",
      label: "a definitive signed-out session without an optional link field",
      session: { authenticated: false },
    },
    {
      expectedKind: "principal-changed",
      label: "a different authenticated WorkOS principal",
      session: {
        authenticated: true,
        provider: "workos",
        mode: "workos",
        account_user_id: "account-b",
        user: { id: "user-b", email: "other@example.org" },
        bisque_linked: false,
      },
    },
  ] as const)("classifies $label after exactly one unlink", async ({
    expectedKind,
    session,
  }) => {
    const currentSession: BisqueAuthSessionResponse = {
      authenticated: true,
      provider: "workos",
      mode: "workos",
      account_user_id: "account-a",
      user: { id: "user-a", email: "researcher@example.org" },
      bisque_linked: true,
    };
    const currentRecoveryScope = noteRecoveryScopeFromSession(currentSession);
    const apiClient = {
      getBisqueSession: vi.fn(),
      unlinkBisqueAccount: vi.fn().mockResolvedValue(session),
    } as unknown as ApiClient;

    const result = await requestAuthenticatedBisqueUnlink(
      apiClient,
      currentRecoveryScope
    );

    expect(result.kind).toBe(expectedKind);
    expect(apiClient.unlinkBisqueAccount).toHaveBeenCalledTimes(1);
    expect(apiClient.getBisqueSession).not.toHaveBeenCalled();
  });

  it("reconciles an older unlink response with the current session", async () => {
    const reconciledSession = {
      authenticated: true,
      provider: "workos",
      mode: "workos",
      account_user_id: "account-a",
      bisque_linked: false,
    } as const;
    const apiClient = {
      unlinkBisqueAccount: vi.fn().mockResolvedValue({
        ...reconciledSession,
        bisque_linked: undefined,
      }),
      getBisqueSession: vi.fn().mockResolvedValue(reconciledSession),
    } as unknown as ApiClient;

    await expect(
      requestAuthenticatedBisqueUnlink(apiClient, "v1:account:account-a")
    ).resolves.toEqual({
      kind: "session-preserved",
      recoveryScope: "v1:account:account-a",
      session: reconciledSession,
    });
    expect(apiClient.unlinkBisqueAccount).toHaveBeenCalledTimes(1);
    expect(apiClient.getBisqueSession).toHaveBeenCalledTimes(1);
  });

  it("fails closed when neither response proves the active principal", async () => {
    const malformedSession = {
      authenticated: "true",
      provider: "workos",
      mode: "workos",
      bisque_linked: false,
    };
    const apiClient = {
      unlinkBisqueAccount: vi.fn().mockResolvedValue(malformedSession),
      getBisqueSession: vi.fn().mockResolvedValue(malformedSession),
    } as unknown as ApiClient;

    await expect(
      requestAuthenticatedBisqueUnlink(apiClient, "v1:account:account-a")
    ).resolves.toEqual({ kind: "indeterminate" });
    expect(apiClient.unlinkBisqueAccount).toHaveBeenCalledTimes(1);
    expect(apiClient.getBisqueSession).toHaveBeenCalledTimes(1);
  });

  it("fails closed when provider and mode disagree about WorkOS authority", async () => {
    const contradictorySession = {
      authenticated: true,
      provider: "local",
      mode: "workos",
      account_user_id: "account-a",
      bisque_linked: false,
    };
    const apiClient = {
      unlinkBisqueAccount: vi.fn().mockResolvedValue(contradictorySession),
      getBisqueSession: vi.fn().mockResolvedValue(contradictorySession),
    } as unknown as ApiClient;

    await expect(
      requestAuthenticatedBisqueUnlink(apiClient, "v1:account:account-a")
    ).resolves.toEqual({ kind: "indeterminate" });
    expect(apiClient.unlinkBisqueAccount).toHaveBeenCalledTimes(1);
    expect(apiClient.getBisqueSession).toHaveBeenCalledTimes(1);
  });

  it("does not hide an unlink request failure behind session reconciliation", async () => {
    const failure = new Error("unlink unavailable");
    const apiClient = {
      unlinkBisqueAccount: vi.fn().mockRejectedValue(failure),
      getBisqueSession: vi.fn(),
    } as unknown as ApiClient;

    await expect(
      requestAuthenticatedBisqueUnlink(apiClient, "v1:account:account-a")
    ).rejects.toBe(failure);
    expect(apiClient.unlinkBisqueAccount).toHaveBeenCalledTimes(1);
    expect(apiClient.getBisqueSession).not.toHaveBeenCalled();
  });
});

describe("owner-bound WorkOS unlink orchestration", () => {
  const scopeA = "v1:account:account-a";
  const scopeB = "v1:account:account-b";
  const prepared = { recoveryScope: scopeA, captureScope: scopeA };
  const session = {
    authenticated: true,
    provider: "workos",
    mode: "workos",
    account_user_id: "account-a",
    bisque_linked: false,
  } as const;
  const deferred = <T,>() => {
    let resolve!: (value: T) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<T>((onResolve, onReject) => {
      resolve = onResolve;
      reject = onReject;
    });
    return { promise, resolve, reject };
  };

  it("applies a confirmed unlink only after the unchanged owner's writer finishes", async () => {
    const flush = deferred<typeof prepared>();
    const apiClient = {
      unlinkBisqueAccount: vi.fn().mockResolvedValue(session),
      getBisqueSession: vi.fn(),
    };
    const applyResult = vi.fn();
    const transition = performAuthenticatedBisqueUnlink({
      apiClient,
      ownerScope: scopeA,
      currentOwnerScope: () => scopeA,
      prepareOwnerState: () => flush.promise,
      applyResult,
    });

    expect(apiClient.unlinkBisqueAccount).not.toHaveBeenCalled();
    expect(applyResult).not.toHaveBeenCalled();
    flush.resolve(prepared);
    await transition;
    expect(apiClient.unlinkBisqueAccount).toHaveBeenCalledTimes(1);
    expect(applyResult).toHaveBeenCalledExactlyOnceWith(
      { kind: "session-preserved", recoveryScope: scopeA, session },
      prepared
    );
  });

  it("does not send unlink after the account changes during the Note flush", async () => {
    let currentOwner = scopeA;
    const flush = deferred<typeof prepared>();
    const apiClient = { unlinkBisqueAccount: vi.fn(), getBisqueSession: vi.fn() };
    const applyResult = vi.fn();
    const transition = performAuthenticatedBisqueUnlink({
      apiClient,
      ownerScope: scopeA,
      currentOwnerScope: () => currentOwner,
      prepareOwnerState: () => flush.promise,
      applyResult,
    });

    currentOwner = scopeB;
    flush.resolve(prepared);
    await transition;
    expect(apiClient.unlinkBisqueAccount).not.toHaveBeenCalled();
    expect(applyResult).not.toHaveBeenCalled();
  });

  it.each([
    ["same-owner unlink", session],
    ["sign-out", { authenticated: false }],
    ["principal change", { ...session, account_user_id: "account-b" }],
  ] as const)("discards a delayed %s receipt after the account changes", async (_label, receipt) => {
    let currentOwner = scopeA;
    const request = deferred<BisqueAuthSessionResponse>();
    const requestStarted = deferred<void>();
    const apiClient = {
      unlinkBisqueAccount: vi.fn(() => {
        requestStarted.resolve();
        return request.promise;
      }),
      getBisqueSession: vi.fn(),
    };
    const applyResult = vi.fn();
    const transition = performAuthenticatedBisqueUnlink({
      apiClient,
      ownerScope: scopeA,
      currentOwnerScope: () => currentOwner,
      prepareOwnerState: async () => prepared,
      applyResult,
    });

    await requestStarted.promise;
    currentOwner = scopeB;
    request.resolve(receipt);
    await transition;
    expect(apiClient.unlinkBisqueAccount).toHaveBeenCalledTimes(1);
    expect(applyResult).not.toHaveBeenCalled();
  });

  it.each(["success", "failure"])("discards stale reconciliation %s", async (outcome) => {
    let currentOwner = scopeA;
    const reconciliation = deferred<BisqueAuthSessionResponse>();
    const reconciliationStarted = deferred<void>();
    const apiClient = {
      unlinkBisqueAccount: vi.fn().mockResolvedValue({ ...session, bisque_linked: undefined }),
      getBisqueSession: vi.fn(() => {
        reconciliationStarted.resolve();
        return reconciliation.promise;
      }),
    };
    const applyResult = vi.fn();
    const transition = performAuthenticatedBisqueUnlink({
      apiClient,
      ownerScope: scopeA,
      currentOwnerScope: () => currentOwner,
      prepareOwnerState: async () => prepared,
      applyResult,
    });

    await reconciliationStarted.promise;
    currentOwner = scopeB;
    if (outcome === "success") reconciliation.resolve(session);
    else reconciliation.reject(new Error("session unavailable"));
    await transition;
    expect(applyResult).not.toHaveBeenCalled();
  });

  it.each(["flush", "request"])("suppresses an old account's delayed %s error", async (stage) => {
    let currentOwner = scopeA;
    const pending = deferred<typeof prepared & BisqueAuthSessionResponse>();
    const requestStarted = deferred<void>();
    const apiClient = {
      unlinkBisqueAccount: vi.fn(() => {
        requestStarted.resolve();
        return pending.promise;
      }),
      getBisqueSession: vi.fn(),
    };
    const applyResult = vi.fn();
    const transition = performAuthenticatedBisqueUnlink({
      apiClient,
      ownerScope: scopeA,
      currentOwnerScope: () => currentOwner,
      prepareOwnerState: stage === "flush" ? () => pending.promise : async () => prepared,
      applyResult,
    });

    if (stage === "request") await requestStarted.promise;
    currentOwner = scopeB;
    pending.reject(new Error("account-a request failed"));
    await expect(transition).resolves.toBeUndefined();
    expect(applyResult).not.toHaveBeenCalled();
  });

  it("preserves retry errors while the initiating account is still current", async () => {
    const failure = new Error("unlink unavailable");
    const apiClient = {
      unlinkBisqueAccount: vi.fn().mockRejectedValue(failure),
      getBisqueSession: vi.fn(),
    };
    const applyResult = vi.fn();
    await expect(performAuthenticatedBisqueUnlink({
      apiClient,
      ownerScope: scopeA,
      currentOwnerScope: () => scopeA,
      prepareOwnerState: async () => prepared,
      applyResult,
    })).rejects.toBe(failure);
    expect(applyResult).not.toHaveBeenCalled();
  });
});
