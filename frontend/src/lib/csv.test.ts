import { describe, expect, it } from "vitest";

import { looksNumeric, parseCsv, sniffDelimiter } from "./csv";

describe("parseCsv", () => {
  it("parses a simple comma-delimited table", () => {
    const { rows } = parseCsv("id,name\n1,alpha\n2,beta\n", ",");
    expect(rows).toEqual([
      ["id", "name"],
      ["1", "alpha"],
      ["2", "beta"],
    ]);
  });

  it("preserves embedded delimiters, quotes, and newlines in quoted fields", () => {
    const { rows } = parseCsv('id,note\n1,"a, b ""q"" \nsecond"\n2,plain\n', ",");
    expect(rows).toEqual([
      ["id", "note"],
      ["1", 'a, b "q" \nsecond'],
      ["2", "plain"],
    ]);
  });

  it("drops a partial last row when asked (truncated head)", () => {
    const full = parseCsv("a,b\n1,2\n3,4", ",", false);
    expect(full.rows).toHaveLength(3);
    expect(full.partialLastRow).toBe(false);

    const trimmed = parseCsv("a,b\n1,2\n3,4", ",", true);
    expect(trimmed.rows).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
    expect(trimmed.partialLastRow).toBe(true);
  });

  it("handles tab delimiter", () => {
    const { rows } = parseCsv("a\tb\n1\t2\n", "\t");
    expect(rows).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });
});

describe("sniffDelimiter", () => {
  it("prefers the consistently-counted delimiter", () => {
    expect(sniffDelimiter("a,b,c\n1,2,3\n4,5,6")).toBe(",");
    expect(sniffDelimiter("a\tb\tc\n1\t2\t3")).toBe("\t");
    expect(sniffDelimiter("a;b;c\n1;2;3")).toBe(";");
  });
  it("defaults to comma when there is no delimiter", () => {
    expect(sniffDelimiter("single\ncolumn")).toBe(",");
  });
});

describe("looksNumeric", () => {
  it("recognizes numbers, decimals, and percentages", () => {
    expect(looksNumeric("42")).toBe(true);
    expect(looksNumeric("-3.14")).toBe(true);
    expect(looksNumeric("1,234")).toBe(true);
    expect(looksNumeric("95%")).toBe(true);
    expect(looksNumeric("1e9")).toBe(true);
  });
  it("rejects non-numbers", () => {
    expect(looksNumeric("alpha")).toBe(false);
    expect(looksNumeric("")).toBe(false);
    expect(looksNumeric("2026-05-12")).toBe(false);
  });
});
