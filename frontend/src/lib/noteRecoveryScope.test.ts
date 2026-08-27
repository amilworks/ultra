import { describe, expect, it } from "vitest";

import { noteRecoveryScopeFromSession } from "./noteRecoveryScope";

describe("noteRecoveryScopeFromSession", () => {
  it("uses immutable Ultra identity rather than mutable session metadata", () => {
    const before = noteRecoveryScopeFromSession({
      authenticated: true,
      mode: "workos",
      provider: "workos",
      username: "scientist@example.org",
      account_user_id: "user_123",
      user: { id: "workos_123", org_id: "org_1" },
    });
    const afterLink = noteRecoveryScopeFromSession({
      authenticated: true,
      mode: "bisque",
      provider: "local",
      username: "linked-bisque-name",
      account_user_id: "user_123",
      user: { id: "different-principal", org_id: "org_2" },
    });

    expect(afterLink).toBe(before);
  });

  it("separates account ids and preserves opaque identifier case", () => {
    const base = {
      authenticated: true,
      mode: "workos" as const,
      provider: "workos" as const,
      account_user_id: "user_123",
      user: { org_id: "org_1" },
    };

    expect(noteRecoveryScopeFromSession({ ...base, account_user_id: "user_456" })).not.toBe(
      noteRecoveryScopeFromSession(base)
    );
    expect(
      noteRecoveryScopeFromSession({
        authenticated: true,
        mode: "workos",
        provider: "workos",
        account_user_id: "User_123",
      })
    ).not.toBe(noteRecoveryScopeFromSession(base));
  });

  it("namespaces principal-id fallback by provider", () => {
    const workos = noteRecoveryScopeFromSession({
      authenticated: true,
      mode: "workos",
      provider: "workos",
      user: { id: "Principal_A" },
    });
    const local = noteRecoveryScopeFromSession({
      authenticated: true,
      mode: "bisque",
      provider: "local",
      user: { id: "Principal_A" },
    });
    expect(workos).not.toBe(local);
    expect(workos).toContain("Principal_A");
  });

  it("returns null for unauthenticated or display-identity-only sessions", () => {
    expect(noteRecoveryScopeFromSession({ authenticated: false })).toBeNull();
    expect(noteRecoveryScopeFromSession({ authenticated: true })).toBeNull();
    expect(
      noteRecoveryScopeFromSession({
        authenticated: true,
        username: "display-name",
        account_email: "email@example.org",
      })
    ).toBeNull();
  });
});
