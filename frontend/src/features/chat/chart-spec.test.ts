import { describe, expect, it } from "vitest";

import { chartTokenVar, ChartSpecError, parseChartSpec } from "./chart-spec";

const validLine = JSON.stringify({
  type: "line",
  title: "Weekly runs",
  x: "week",
  series: [
    { key: "runs", label: "Total runs" },
    { key: "useful", label: "Useful", colorIndex: 3 },
  ],
  data: [
    { week: "W1", runs: 120, useful: 88 },
    { week: "W2", runs: 143, useful: 101 },
  ],
});

describe("parseChartSpec — happy path", () => {
  it("parses a valid line spec and assigns brand tokens", () => {
    const spec = parseChartSpec(validLine);
    expect(spec.type).toBe("line");
    expect(spec.x).toBe("week");
    expect(spec.series).toHaveLength(2);
    expect(spec.series[0].colorIndex).toBe(1); // defaults to position
    expect(spec.series[1].colorIndex).toBe(3); // explicit slot
    expect(spec.data).toHaveLength(2);
    expect(spec.stacked).toBe(false);
  });

  it("defaults label to key when omitted", () => {
    const spec = parseChartSpec(
      JSON.stringify({ type: "bar", x: "m", series: [{ key: "v" }], data: [{ m: "a", v: 1 }] }),
    );
    expect(spec.series[0].label).toBe("v");
  });

  it("wraps colorIndex into 1..8", () => {
    const spec = parseChartSpec(
      JSON.stringify({ type: "bar", x: "m", series: [{ key: "v", colorIndex: 9 }], data: [{ m: "a", v: 1 }] }),
    );
    expect(spec.series[0].colorIndex).toBe(1);
  });
});

describe("chartTokenVar", () => {
  it("only ever produces a --chart-N token", () => {
    expect(chartTokenVar(2)).toBe("var(--chart-2)");
    expect(chartTokenVar(99)).toBe("var(--chart-8)"); // clamped
    expect(chartTokenVar(0)).toBe("var(--chart-1)"); // clamped
  });
});

describe("parseChartSpec — rejects (security + robustness)", () => {
  const reject = (json: string) => expect(() => parseChartSpec(json)).toThrow(ChartSpecError);

  it("rejects malformed JSON", () => reject("{not json"));
  it("rejects a non-object spec", () => reject("[1,2,3]"));
  it("rejects an unknown chart type", () =>
    reject(JSON.stringify({ type: "sankey", x: "a", series: [{ key: "b" }], data: [{ a: 1, b: 2 }] })));

  it("rejects a raw color string (only colorIndex allowed)", () => {
    reject(
      JSON.stringify({
        type: "line",
        x: "a",
        series: [{ key: "b", colorIndex: "red; } body{display:none}" }],
        data: [{ a: 1, b: 2 }],
      }),
    );
  });

  it("rejects a non-identifier series key (CSS-injection guard)", () => {
    reject(
      JSON.stringify({
        type: "line",
        x: "a",
        series: [{ key: "b; } :root{--x:url(//evil)}" }],
        data: [{ a: 1 }],
      }),
    );
  });

  it("rejects a non-identifier x key", () =>
    reject(JSON.stringify({ type: "line", x: "a b", series: [{ key: "v" }], data: [{ "a b": 1, v: 2 }] })));

  it("rejects too many series (> 8)", () => {
    const series = Array.from({ length: 9 }, (_, i) => ({ key: `s${i}` }));
    const row: Record<string, number> = { x: 0 };
    series.forEach((s) => (row[s.key] = 1));
    reject(JSON.stringify({ type: "line", x: "x", series, data: [row] }));
  });

  it("rejects too many rows (> 1000)", () => {
    const data = Array.from({ length: 1001 }, (_, i) => ({ x: i, v: i }));
    reject(JSON.stringify({ type: "line", x: "x", series: [{ key: "v" }], data }));
  });

  it("rejects a series key not present in data", () =>
    reject(JSON.stringify({ type: "line", x: "x", series: [{ key: "missing" }], data: [{ x: 1 }] })));

  it("rejects the x key not present in data", () =>
    reject(JSON.stringify({ type: "line", x: "week", series: [{ key: "v" }], data: [{ v: 1 }] })));

  it("rejects a non-finite numeric value", () =>
    // Raw JSON: JSON.parse("1e999") -> Infinity (JSON.stringify(Infinity) would be null).
    reject('{"type":"line","x":"x","series":[{"key":"v"}],"data":[{"x":1,"v":1e999}]}'));

  it("rejects an empty series array", () =>
    reject(JSON.stringify({ type: "line", x: "x", series: [], data: [{ x: 1 }] })));

  it("rejects an object value in a data cell", () =>
    reject(JSON.stringify({ type: "line", x: "x", series: [{ key: "v" }], data: [{ x: 1, v: { a: 1 } }] })));
});
