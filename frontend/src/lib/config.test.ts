import { describe, expect, it } from "vitest";

import { DEFAULT_BISQUE_BROWSER_URL } from "./config";

describe("frontend config", () => {
  it("defaults BisQue browser links to the production client_service", () => {
    expect(DEFAULT_BISQUE_BROWSER_URL).toBe(
      "https://bisque2.ece.ucsb.edu/client_service/"
    );
  });
});
