// Dependency-free, quote-aware CSV/TSV parsing for the bounded head the viewer
// holds in memory (gzip heads + client fallback). The server endpoint is the
// primary path for plain CSV; this handles the cases the server cannot page.

export type ParsedCsv = {
  rows: string[][];
  /** True when the final line had no trailing newline and may be incomplete. */
  partialLastRow: boolean;
};

// parseCsv implements RFC4180-ish parsing: quoted fields may contain the
// delimiter, newlines, and escaped quotes (""). When dropPartial is true the
// final row is dropped if the input did not end on a row boundary (a truncated
// head), so a half-read record is never shown.
export function parseCsv(text: string, delimiter: string, dropPartial = false): ParsedCsv {
  const delim = delimiter || ",";
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let inQuotes = false;
  let sawAny = false;
  let endedOnNewline = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    endedOnNewline = false;
    if (inQuotes) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += char;
      }
      continue;
    }
    if (char === '"') {
      inQuotes = true;
      sawAny = true;
      continue;
    }
    if (char === delim) {
      row.push(field);
      field = "";
      sawAny = true;
      continue;
    }
    if (char === "\n" || char === "\r") {
      if (char === "\r" && text[i + 1] === "\n") {
        i += 1;
      }
      row.push(field);
      rows.push(row);
      field = "";
      row = [];
      sawAny = false;
      endedOnNewline = true;
      continue;
    }
    field += char;
    sawAny = true;
  }

  if (sawAny || field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  let partialLastRow = false;
  if (dropPartial && !endedOnNewline && rows.length > 0) {
    rows.pop();
    partialLastRow = true;
  }
  return { rows, partialLastRow };
}

const DELIMITER_CANDIDATES = [",", "\t", ";", "|"];

// sniffDelimiter picks the most consistent delimiter across the first lines of a
// sample (constant field count per line wins), defaulting to comma.
export function sniffDelimiter(sample: string): string {
  const lines = sample.split("\n").slice(0, 20).filter((line) => line.trim().length > 0);
  if (lines.length === 0) {
    return ",";
  }
  let best = ",";
  let bestScore = -1;
  for (const candidate of DELIMITER_CANDIDATES) {
    const counts = lines.map((line) => line.split(candidate).length - 1);
    const max = Math.max(...counts);
    if (max === 0) {
      continue;
    }
    const min = Math.min(...counts);
    const sum = counts.reduce((total, value) => total + value, 0);
    const score = sum + (min === max ? 1000 : 0);
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }
  return best;
}

// looksNumeric reports whether a cell should be right-aligned as a number.
export function looksNumeric(value: string): boolean {
  const trimmed = value.trim();
  if (trimmed === "") {
    return false;
  }
  return /^[-+]?(\d{1,3}(,\d{3})*|\d+)(\.\d+)?([eE][-+]?\d+)?%?$/.test(trimmed);
}
