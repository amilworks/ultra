// Strict validator for agent-emitted chart specs (the ```chart fenced block).
//
// SECURITY: agent output is prompt-injectable (the agent reads documents, tool
// results, and uploads). This validator is the trust boundary between that
// output and the recharts renderer. It is reject-by-default and, critically,
// the agent CANNOT supply raw color strings — series colors are assigned by
// index from the brand --chart-* tokens, so nothing hostile can reach the
// ChartStyle CSS sink (chart.tsx). Labels/values are rendered by React (escaped).

export const CHART_TYPES = ["line", "bar", "area", "scatter", "pie"] as const;
export type ChartType = (typeof CHART_TYPES)[number];

/** A validated, safe-to-render chart spec. */
export interface ChartSpec {
  type: ChartType;
  title?: string;
  /** Category / x-axis key (also the label key for pie). */
  x: string;
  /** 1..MAX_SERIES value series; colorIndex is a brand token slot 1..8. */
  series: Array<{ key: string; label: string; colorIndex: number }>;
  /** Bounded row data; values are numbers or short strings. */
  data: Array<Record<string, number | string>>;
  stacked: boolean;
}

// DoS / sanity bounds — a chat chart is a summary, not a data dump.
const MAX_ROWS = 1000;
const MAX_SERIES = 8; // == number of brand --chart-* tokens
const MAX_KEYS = 24;
const MAX_STR = 120; // cap any string cell / label / title length
const CHART_TOKENS = 8;

export class ChartSpecError extends Error {}

function str(v: unknown, field: string, max = MAX_STR): string {
  if (typeof v !== "string") throw new ChartSpecError(`${field} must be a string`);
  if (v.length > max) throw new ChartSpecError(`${field} too long`);
  return v;
}

// Series/x keys become CSS custom-property names (`--color-<key>`), so they
// must be plain identifiers — this is a second guard on the ChartStyle sink.
const IDENT = /^[A-Za-z_][A-Za-z0-9_-]{0,63}$/;
function ident(v: unknown, field: string): string {
  const s = str(v, field, 64);
  if (!IDENT.test(s)) {
    throw new ChartSpecError(`${field} must be an identifier ([A-Za-z_][A-Za-z0-9_-]*)`);
  }
  return s;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * Parse + validate a raw ```chart block. Throws ChartSpecError on anything
 * malformed, oversized, or unexpected. Never returns partially-trusted data.
 */
export function parseChartSpec(source: string): ChartSpec {
  let raw: unknown;
  try {
    raw = JSON.parse(source);
  } catch {
    throw new ChartSpecError("not valid JSON");
  }
  if (!isPlainObject(raw)) throw new ChartSpecError("spec must be an object");

  const type = raw.type;
  if (typeof type !== "string" || !CHART_TYPES.includes(type as ChartType)) {
    throw new ChartSpecError(`type must be one of ${CHART_TYPES.join(", ")}`);
  }

  const x = ident(raw.x, "x");

  if (!Array.isArray(raw.series) || raw.series.length < 1) {
    throw new ChartSpecError("series must be a non-empty array");
  }
  if (raw.series.length > MAX_SERIES) {
    throw new ChartSpecError(`too many series (max ${MAX_SERIES})`);
  }
  const seenKeys = new Set<string>();
  const series = raw.series.map((s, i) => {
    if (!isPlainObject(s)) throw new ChartSpecError(`series[${i}] must be an object`);
    const key = ident(s.key, `series[${i}].key`);
    if (key === x) throw new ChartSpecError(`series[${i}].key collides with x`);
    if (seenKeys.has(key)) throw new ChartSpecError(`duplicate series key "${key}"`);
    seenKeys.add(key);
    const label = s.label === undefined ? key : str(s.label, `series[${i}].label`);
    // colorIndex is a BRAND TOKEN SLOT (1..8), never a raw color string.
    let colorIndex = i + 1;
    if (s.colorIndex !== undefined) {
      if (typeof s.colorIndex !== "number" || !Number.isInteger(s.colorIndex)) {
        throw new ChartSpecError(`series[${i}].colorIndex must be an integer 1..${CHART_TOKENS}`);
      }
      colorIndex = ((s.colorIndex - 1) % CHART_TOKENS + CHART_TOKENS) % CHART_TOKENS + 1;
    }
    return { key, label, colorIndex };
  });

  if (!Array.isArray(raw.data) || raw.data.length < 1) {
    throw new ChartSpecError("data must be a non-empty array");
  }
  if (raw.data.length > MAX_ROWS) {
    throw new ChartSpecError(`too many rows (max ${MAX_ROWS})`);
  }
  const data = raw.data.map((row, i) => {
    if (!isPlainObject(row)) throw new ChartSpecError(`data[${i}] must be an object`);
    const keys = Object.keys(row);
    if (keys.length > MAX_KEYS) throw new ChartSpecError(`data[${i}] has too many keys`);
    const out: Record<string, number | string> = {};
    for (const k of keys) {
      const v = row[k];
      if (typeof v === "number") {
        if (!Number.isFinite(v)) throw new ChartSpecError(`data[${i}].${k} is not finite`);
        out[k] = v;
      } else if (typeof v === "string") {
        out[k] = v.length > MAX_STR ? v.slice(0, MAX_STR) : v;
      } else if (v === null || v === undefined) {
        // allow gaps
      } else {
        throw new ChartSpecError(`data[${i}].${k} must be a number or string`);
      }
    }
    return out;
  });

  // Every series key and the x key must appear in the data.
  const present = new Set<string>();
  for (const row of data) for (const k of Object.keys(row)) present.add(k);
  if (!present.has(x)) throw new ChartSpecError(`x key "${x}" not present in data`);
  for (const s of series) {
    if (!present.has(s.key)) throw new ChartSpecError(`series key "${s.key}" not present in data`);
  }

  return {
    type: type as ChartType,
    title: raw.title === undefined ? undefined : str(raw.title, "title"),
    x,
    series,
    data,
    stacked: raw.stacked === true,
  };
}

/** The CSS token for a brand chart slot (1..8). Only ever `var(--chart-N)`. */
export function chartTokenVar(colorIndex: number): string {
  const n = Math.min(CHART_TOKENS, Math.max(1, Math.trunc(colorIndex)));
  return `var(--chart-${n})`;
}
