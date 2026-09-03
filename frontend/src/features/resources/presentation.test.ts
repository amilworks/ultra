import { describe, expect, it } from "vitest";

import {
  derivePastedTitle,
  findQueryMatches,
  groupResourcesByDateSection,
  isPastedTextName,
  resourceDisplayName,
} from "./presentation";

describe("resourceDisplayName", () => {
  it("uses the original name when it is not an internal path", () => {
    expect(
      resourceDisplayName({ original_name: "scan.tif", file_id: "abc" })
    ).toBe("scan.tif");
  });
});

describe("isPastedTextName", () => {
  it("matches the chat paste naming scheme, with and without milliseconds", () => {
    expect(isPastedTextName("pasted-2026-08-29-004747-474.txt")).toBe(true);
    expect(isPastedTextName("pasted-2026-08-29-004747.txt")).toBe(true);
  });

  it("does not match ordinary uploads or pasted images", () => {
    expect(isPastedTextName("fused.ply")).toBe(false);
    expect(isPastedTextName("pasted-notes.txt")).toBe(false);
    expect(isPastedTextName("pasted-2026-08-29-004747-474.png")).toBe(false);
  });
});

describe("derivePastedTitle", () => {
  it("takes the first content line and strips markdown decoration", () => {
    const head =
      "\n### 1. Definite sign error: **not right**\nYou define\n$$\nQ=I-2nn^T\n$$\n";
    expect(derivePastedTitle(head)).toBe("1. Definite sign error: not right");
  });

  it("skips fences, dividers, and bare TeX lines", () => {
    const head = "$$\n\\tfrac12 x\n---\nThe main result is correct.\n";
    expect(derivePastedTitle(head)).toBe("The main result is correct.");
  });

  it("truncates long lines at a word boundary with an ellipsis", () => {
    const long =
      "The main result is correct, but I would not submit the writeup exactly as it stands today because";
    const title = derivePastedTitle(long);
    expect(title).not.toBeNull();
    expect(title!.length).toBeLessThanOrEqual(73);
    expect(title!.endsWith("…")).toBe(true);
    expect(title).not.toContain("  ");
  });

  it("returns null when nothing usable appears in the opening lines", () => {
    expect(derivePastedTitle("$$\n\\begin{bmatrix}1\\end{bmatrix}\n$$")).toBeNull();
    expect(derivePastedTitle("")).toBeNull();
  });
});

describe("groupResourcesByDateSection", () => {
  const now = new Date(2026, 7, 29, 12, 0, 0); // Aug 29 2026, local time
  const at = (iso: string) => ({ created_at: iso });

  it("buckets into Today / Yesterday / Last 7 days / month labels", () => {
    const sections = groupResourcesByDateSection(
      [
        at(new Date(2026, 7, 29, 0, 47).toISOString()),
        at(new Date(2026, 7, 28, 23, 0).toISOString()),
        at(new Date(2026, 7, 24, 9, 0).toISOString()),
        at(new Date(2026, 7, 2, 9, 0).toISOString()),
        at(new Date(2025, 11, 20, 9, 0).toISOString()),
      ],
      now
    );
    expect(sections.map((section) => section.label)).toEqual([
      "Today",
      "Yesterday",
      "Last 7 days",
      "August",
      "December 2025",
    ]);
  });

  it("merges adjacent same-bucket items and preserves order", () => {
    const first = at(new Date(2026, 7, 29, 1, 0).toISOString());
    const second = at(new Date(2026, 7, 29, 0, 30).toISOString());
    const sections = groupResourcesByDateSection([first, second], now);
    expect(sections).toHaveLength(1);
    expect(sections[0].items).toEqual([first, second]);
  });

  it("labels unparseable dates instead of throwing", () => {
    const sections = groupResourcesByDateSection([at("not-a-date")], now);
    expect(sections[0].label).toBe("Undated");
  });
});

describe("findQueryMatches", () => {
  it("finds case-insensitive occurrences", () => {
    expect(findQueryMatches("Fused.PLY", "ply")).toEqual([{ start: 6, end: 9 }]);
  });

  it("finds repeated occurrences without overlap", () => {
    expect(findQueryMatches("aaaa", "aa")).toEqual([
      { start: 0, end: 2 },
      { start: 2, end: 4 },
    ]);
  });

  it("returns nothing for an empty or whitespace query", () => {
    expect(findQueryMatches("anything", "  ")).toEqual([]);
  });
});
