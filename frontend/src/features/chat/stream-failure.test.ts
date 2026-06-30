import { describe, expect, it } from "vitest";

import { classifyStreamFailure, composeStreamFailureReason } from "./stream-failure";

describe("classifyStreamFailure", () => {
  it("flags auth failures as non-retryable with a re-auth headline", () => {
    for (const status of [401, 403]) {
      const c = classifyStreamFailure(status, "request failed with status " + status);
      expect(c.category).toBe("auth");
      expect(c.retryable).toBe(false);
      expect(c.headline.toLowerCase()).toContain("sign in");
    }
  });

  it("flags rate limiting (429 or text) as retryable", () => {
    expect(classifyStreamFailure(429, "boom").category).toBe("rate_limited");
    expect(classifyStreamFailure(undefined, "Rate limit exceeded").category).toBe("rate_limited");
    expect(classifyStreamFailure(429, "boom").retryable).toBe(true);
  });

  it("treats network/transport drops as retryable transient transport", () => {
    for (const detail of ["Load failed", "Failed to fetch", "The network connection was lost", "terminated"]) {
      const c = classifyStreamFailure(undefined, detail);
      expect(c.category).toBe("transient_transport");
      expect(c.retryable).toBe(true);
      expect(c.headline.toLowerCase()).toContain("kept above");
    }
  });

  it("treats 5xx as a retryable server run failure", () => {
    const c = classifyStreamFailure(503, "service unavailable");
    expect(c.category).toBe("run_failed");
    expect(c.retryable).toBe(true);
  });

  it("defaults unknown failures: retryable when no or 5xx status, not for a clean 4xx", () => {
    expect(classifyStreamFailure(undefined, "weird").retryable).toBe(true);
    expect(classifyStreamFailure(400, "bad request").retryable).toBe(false);
    expect(classifyStreamFailure(400, "bad request").category).toBe("unknown");
  });
});

describe("composeStreamFailureReason", () => {
  it("leads with the calm headline and appends the technical detail", () => {
    const c = classifyStreamFailure(503, "Internal Server Error");
    const reason = composeStreamFailureReason(c, "Internal Server Error");
    expect(reason.startsWith(c.headline)).toBe(true);
    expect(reason).toContain("(Internal Server Error)");
  });

  it("omits an empty or redundant detail", () => {
    const c = classifyStreamFailure(401, "");
    expect(composeStreamFailureReason(c, "")).toBe(c.headline);
  });
});
