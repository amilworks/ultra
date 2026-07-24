import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const readSource = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

const stylesSource = readSource("src/styles.css");
const typographySource = readSource("src/typography.css");

const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const blockBody = (source: string, selector: string, requiredText?: string): string => {
  const selectorMatches = [
    ...source.matchAll(new RegExp(`^${escapeRegExp(selector)}\\s*\\{`, "gm")),
  ];
  for (const selectorMatch of selectorMatches) {
    const selectorStart = selectorMatch.index;
    const bodyStart = source.indexOf("{", selectorStart);
    const bodyEnd = source.indexOf("}", bodyStart);
    const body = source.slice(bodyStart + 1, bodyEnd);
    if (!requiredText || body.includes(requiredText)) {
      return body;
    }
  }
  throw new Error(
    `Missing CSS selector ${selector}${requiredText ? ` containing ${requiredText}` : ""}`
  );
};

const groupedBlockBody = (source: string, selectorAnchor: string): string => {
  const selectorMatch = new RegExp(`^${escapeRegExp(selectorAnchor)}`, "m").exec(source);
  expect(selectorMatch, `Missing CSS selector anchor ${selectorAnchor}`).not.toBeNull();
  const selectorStart = selectorMatch?.index ?? -1;
  const bodyStart = source.indexOf("{", selectorStart);
  const bodyEnd = source.indexOf("}", bodyStart);
  return source.slice(bodyStart + 1, bodyEnd);
};

const variables = (body: string): Map<string, string> =>
  new Map(
    [...body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map((match) => [
      match[1],
      match[2].trim().toLowerCase(),
    ])
  );

const relativeLuminance = (hex: string): number => {
  const channels = hex
    .replace("#", "")
    .match(/.{2}/g)
    ?.map((channel) => Number.parseInt(channel, 16) / 255)
    .map((channel) =>
      channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4
    );
  if (!channels || channels.length !== 3) {
    throw new Error(`Expected a six-digit hex color, received ${hex}`);
  }
  return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
};

const contrastRatio = (foreground: string, background: string): number => {
  const foregroundLuminance = relativeLuminance(foreground);
  const backgroundLuminance = relativeLuminance(background);
  return (
    (Math.max(foregroundLuminance, backgroundLuminance) + 0.05) /
    (Math.min(foregroundLuminance, backgroundLuminance) + 0.05)
  );
};

const lightTokens = variables(blockBody(stylesSource, ":root"));
const darkTokenOverrides = variables(blockBody(stylesSource, ".dark", "--bg-page:"));
const darkTokens = new Map([
  ...lightTokens,
  ...darkTokenOverrides,
]);
const typographyLightTokens = variables(blockBody(typographySource, ":root"));
const typographyDarkTokens = new Map([
  ...typographyLightTokens,
  ...variables(blockBody(typographySource, ".dark")),
]);

describe("Ultra semantic blue accent contract", () => {
  it("defines exact light and dark semantic tokens without changing the chart palette", () => {
    expect(Object.fromEntries(lightTokens)).toMatchObject({
      "--ultra-accent-signal": "#1994ff",
      "--ultra-accent-solid": "#006fd6",
      "--ultra-accent-foreground": "#ffffff",
      "--ultra-accent-text": "#0068c9",
      "--ultra-accent-soft": "#e8f4ff",
      "--ultra-accent-border": "#9fd1ff",
      "--activity-heatmap-1": "#b9dcff",
      "--activity-heatmap-2": "#75bcff",
      "--activity-heatmap-3": "#319fff",
      "--activity-heatmap-4": "#087ce8",
      "--status-success": "#047857",
    });
    expect(Object.fromEntries(darkTokenOverrides)).toMatchObject({
      "--ultra-accent-signal": "#1994ff",
      "--ultra-accent-solid": "#006fd6",
      "--ultra-accent-foreground": "#ffffff",
      "--ultra-accent-text": "#67b7ff",
      "--ultra-accent-soft": "#10263a",
      "--ultra-accent-border": "#16496d",
      "--activity-heatmap-1": "#16496d",
      "--activity-heatmap-2": "#13699f",
      "--activity-heatmap-3": "#147fcc",
      "--activity-heatmap-4": "#1994ff",
      "--status-success": "#6ee7b7",
    });

    expect(lightTokens.get("--chart-2")).toBe("oklch(0.559 0.169 41.7)");
    expect(darkTokens.get("--chart-2")).toBe("oklch(0.665 0.169 42.0)");
    expect(stylesSource.match(/--chart-2\s*:/g)).toHaveLength(2);
  });

  it("meets the required contrast and heatmap luminance progression", () => {
    expect(contrastRatio("#006fd6", "#ffffff")).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio("#1994ff", "#111113")).toBeGreaterThanOrEqual(3);

    const lightLuminance = [1, 2, 3, 4].map((level) =>
      relativeLuminance(lightTokens.get(`--activity-heatmap-${level}`) ?? "")
    );
    const darkLuminance = [1, 2, 3, 4].map((level) =>
      relativeLuminance(darkTokens.get(`--activity-heatmap-${level}`) ?? "")
    );
    expect(lightLuminance.every((value, index) => index === 0 || value < lightLuminance[index - 1]))
      .toBe(true);
    expect(darkLuminance.every((value, index) => index === 0 || value > darkLuminance[index - 1]))
      .toBe(true);
  });

  it("moves only the approved non-data UI groups off chart-2", () => {
    const avatarRules = groupedBlockBody(
      stylesSource,
      '.app-sidebar-account-avatar [data-slot="avatar-fallback"],'
    );
    expect(avatarRules).not.toContain("var(--chart-2)");
    expect(avatarRules).toContain("background: var(--ultra-accent-solid)");
    expect(avatarRules).toContain("color: var(--ultra-accent-foreground)");

    const heatmapRules = [1, 2, 3, 4]
      .map((level) => blockBody(stylesSource, `.token-heatmap-cell[data-level="${level}"]`))
      .join("\n");
    expect(heatmapRules).not.toContain("var(--chart-2)");
    for (const level of [1, 2, 3, 4]) {
      expect(heatmapRules).toContain(`var(--activity-heatmap-${level})`);
    }

    const bisqueRules = [
      ".app-settings-bisque-link-status",
      '.app-settings-bisque-link-status svg[data-icon]',
      '.app-settings-bisque-linked-alert[data-slot="alert"]',
      '.app-settings-bisque-linked-alert > svg[data-icon]',
      '.app-settings-bisque-linked-alert [data-slot="alert-action"] [data-slot="button"]',
    ]
      .map((selector) => blockBody(stylesSource, selector))
      .join("\n");
    expect(bisqueRules).not.toContain("var(--chart-2)");
    expect(bisqueRules).toContain("var(--status-success)");

    const runtimeGoodRules = blockBody(stylesSource, ".admin-runtime-good");
    expect(runtimeGoodRules).not.toContain("var(--chart-2)");
    expect(runtimeGoodRules).toContain("var(--status-success)");
  });

  it("preserves representative chart-2 data series and warning and danger semantics", () => {
    for (const relativePath of [
      "src/components/AdminPlatformValue.tsx",
      "src/components/AdminConsole.tsx",
      "src/components/viewer/hdf5/MaterialsHdf5Dashboard.tsx",
      "src/components/viewer/hdf5/Hdf5DatasetPreview.tsx",
    ]) {
      expect(readSource(relativePath), relativePath).toContain('color: "var(--chart-2)"');
    }
    expect(readSource("src/features/chat/chart-spec.ts")).toContain(
      "return `var(--chart-${n})`"
    );

    expect(lightTokens.get("--danger")).toBe("#c62828");
    expect(lightTokens.get("--destructive")).toBe("oklch(0.577 0.245 27.325)");
    expect(darkTokens.get("--destructive")).toBe("oklch(0.704 0.191 22.216)");
    const runtimeWarningRules = blockBody(stylesSource, ".admin-runtime-warn");
    expect(runtimeWarningRules).toContain("var(--chart-4)");
    expect(runtimeWarningRules).not.toContain("var(--status-success)");
  });

  it("uses accessible blue only for the Ultra half of the wordmark", () => {
    expect(typographyLightTokens.get("--brand-wordmark-accent-light")).toBe("#0068c9");
    expect(typographyLightTokens.get("--brand-wordmark-accent-dark")).toBe("#67b7ff");
    expect(typographyDarkTokens.get("--brand-wordmark-accent")).toBe(
      "var(--brand-wordmark-accent-dark)"
    );
    expect(contrastRatio("#0068c9", "#ffffff")).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio("#67b7ff", "#111113")).toBeGreaterThanOrEqual(4.5);

    expect(blockBody(typographySource, ".brand-wordmark__bisque")).not.toContain("color:");
    expect(blockBody(typographySource, ".brand-wordmark__ultra")).toContain(
      "color: var(--brand-wordmark-accent)"
    );
  });
});
