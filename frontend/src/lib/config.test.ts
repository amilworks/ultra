import { describe, expect, it } from "vitest";

import { DEFAULT_BISQUE_BROWSER_URL } from "./config";

describe("frontend config", () => {
  it("defaults BisQue browser links to the current origin client_service", () => {
    expect(DEFAULT_BISQUE_BROWSER_URL).toBe(
      `${window.location.origin}/client_service/`
    );
  });
});
