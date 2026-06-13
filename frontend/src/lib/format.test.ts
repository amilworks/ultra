import { describe, expect, it } from "vitest";

import { formatDurationSeconds, formatTokens } from "./format";

describe("formatTokens", () => {
  it("renders small counts verbatim", () => {
    expect(formatTokens(0)).toBe("0");
    expect(formatTokens(1)).toBe("1");
    expect(formatTokens(999)).toBe("999");
  });

  it("compacts thousands, millions, and billions", () => {
    expect(formatTokens(1500)).toBe("1.5K");
    expect(formatTokens(14570)).toBe("14.6K");
    expect(formatTokens(812304)).toBe("812K");
    expect(formatTokens(1_250_000)).toBe("1.3M");
    expect(formatTokens(19_900_000_000)).toBe("19.9B");
  });

  it("guards against invalid input", () => {
    expect(formatTokens(-50)).toBe("0");
    expect(formatTokens(Number.NaN)).toBe("0");
  });
});

describe("formatDurationSeconds", () => {
  it("renders hours and minutes for long tasks", () => {
    expect(formatDurationSeconds(38580)).toBe("10h 43m");
    expect(formatDurationSeconds(3600)).toBe("1h 0m");
  });

  it("renders minutes and seconds for shorter tasks", () => {
    expect(formatDurationSeconds(125)).toBe("2m 5s");
    expect(formatDurationSeconds(42)).toBe("42s");
  });

  it("renders an em dash for empty or invalid durations", () => {
    expect(formatDurationSeconds(0)).toBe("—");
    expect(formatDurationSeconds(-1)).toBe("—");
    expect(formatDurationSeconds(Number.NaN)).toBe("—");
  });
});
