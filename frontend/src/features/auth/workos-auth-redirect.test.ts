import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const readSource = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

describe("WorkOS hosted auth redirect", () => {
  it("sends unauthenticated WorkOS sessions to AuthKit before rendering local auth", () => {
    const source = readSource("src/App.tsx");
    const localAuthScreenIndex = source.indexOf("<AuthScreen");
    const authBranchStart = source.lastIndexOf(
      'if (authStatus !== "authenticated")',
      localAuthScreenIndex
    );
    const authBranchEnd = source.indexOf("return (", localAuthScreenIndex);
    const authBranch = source.slice(authBranchStart, authBranchEnd);

    expect(authBranch).toContain('authProvider === "workos"');
    expect(authBranch).toContain("<WorkOSRedirectScreen");
    expect(authBranch.indexOf("<WorkOSRedirectScreen")).toBeLessThan(
      authBranch.indexOf("<AuthScreen")
    );
    expect(source).toContain("hostedAuthRedirectAttemptedRef");
    expect(source).not.toContain("Account review needed");
  });

  it("does not auto-redirect denied accounts back into AuthKit", () => {
    const source = readSource("src/App.tsx");
    const redirectEffectStart = source.indexOf("hostedAuthRedirectAttemptedRef.current = true");
    const redirectGuard = source.slice(
      source.lastIndexOf("useEffect", redirectEffectStart),
      redirectEffectStart
    );

    // A pending/disabled account sets authNotice; redirecting again would
    // bounce through AuthKit forever because WorkOS still has a live session.
    expect(redirectGuard).toContain("authNotice");
  });

  it("surfaces account denial notices on the WorkOS redirect screen", () => {
    const source = readSource("src/App.tsx");
    const screensSource = readSource("src/components/auth/AuthShellScreens.tsx");
    const redirectScreenIndex = source.indexOf("<WorkOSRedirectScreen");
    const redirectScreenBlock = source.slice(redirectScreenIndex, redirectScreenIndex + 400);

    expect(redirectScreenBlock).toContain("statusMessage=");
    expect(screensSource).toContain("statusMessage?: string | null");
  });

  it("does not publish blank post-login draft conversation ids into the URL", () => {
    const source = readSource("src/App.tsx");
    const effectMarker = 'authStatus !== "authenticated" || !conversationsHydrated';
    const effectMarkerIndex = source.indexOf(effectMarker);
    const urlEffectStart = source.lastIndexOf("useEffect", effectMarkerIndex);
    const urlEffectEnd = source.indexOf("  const flushConversationSnapshots", urlEffectStart);
    const urlEffect = source.slice(urlEffectStart, urlEffectEnd);

    expect(urlEffect).toContain("shouldExposeConversationInUrl");
    expect(urlEffect).toContain("replaceConversationIdInLocation(");
    // The guard survives the push-history refactor: draft conversations resolve
    // to a null URL target (replace path), never a pushed history entry.
    expect(urlEffect).toContain("? resolvedConversationId");
    expect(urlEffect).toContain(": null");
    expect(urlEffect).toContain("pushConversationIdInLocation(");
  });

  it("replaces a stale requested chat URL before activating a live fallback", () => {
    const source = readSource("src/App.tsx");
    const bootstrapStart = source.indexOf("let missingRequestedConversationId");
    const bootstrapEnd = source.indexOf("// A failed bootstrap", bootstrapStart);
    const bootstrap = source.slice(bootstrapStart, bootstrapEnd);
    const fallbackStart = bootstrap.indexOf("const missingConversationFallback");
    const fallbackEnd = bootstrap.indexOf("setActiveConversationId", fallbackStart);
    const fallbackBlock = bootstrap.slice(fallbackStart, fallbackEnd);

    expect(fallbackBlock).toContain("replaceConversationIdInLocation(");
    expect(fallbackBlock).not.toContain("pushConversationIdInLocation(");
    expect(bootstrap).toContain("return missingConversationFallback.id");
  });

  it("uses accurate stale-chat recovery copy for existing and new fallbacks", () => {
    const source = readSource("src/App.tsx");
    const hydrateStart = source.indexOf("const ensureConversationHydrated");
    const hydrateEnd = source.indexOf("const loadMoreConversations", hydrateStart);
    const hydrate = source.slice(hydrateStart, hydrateEnd);

    expect(hydrate).toContain("existingFallbackConversation");
    expect(hydrate).toContain("MISSING_REQUESTED_CONVERSATION_MESSAGE");
    expect(hydrate).toContain("MISSING_REQUESTED_CONVERSATION_NEW_MESSAGE");
  });

  it("clears authenticated state only after logout or unlink succeeds", () => {
    const source = readSource("src/App.tsx");
    const start = source.indexOf("const leaveAuthenticatedAccount");
    const end = source.indexOf("const logoutBisque", start);
    const departure = source.slice(start, end);
    const request = departure.indexOf("requestAuthenticatedAccountDeparture");
    const failureReturn = departure.indexOf("return;", request);
    const clear = departure.indexOf("clearAuthViewState();", request);

    expect(departure).toContain("enableNoteDraftRecoveryScope(logoutScope)");
    expect(departure).toContain("You’re still signed in; try again.");
    expect(request).toBeGreaterThan(-1);
    expect(failureReturn).toBeGreaterThan(request);
    expect(clear).toBeGreaterThan(failureReturn);
  });
});
