import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const readSource = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

describe("local BisQue auth", () => {
  it("does not ship the legacy Keycloak or OIDC frontend auth path", () => {
    const productionSources = [
      "src/App.tsx",
      "src/components/auth/AuthScreen.tsx",
      "src/lib/api.ts",
      "src/types.ts",
      "scripts/mock-api.mjs",
    ].map(readSource);

    productionSources.forEach((source) => {
      expect(source).not.toMatch(/Keycloak|SSO|OIDC|oidc|auth\/oidc|bisque_oidc|authOidc/i);
    });
  });
});
