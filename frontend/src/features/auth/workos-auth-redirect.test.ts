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
});
