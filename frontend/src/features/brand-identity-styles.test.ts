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

describe("Ultra monochrome identity contract", () => {
  it("defines exact monochrome identity tokens and leaves launch blue unused", () => {
    expect(Object.fromEntries(lightTokens)).toMatchObject({
      "--ultra-launch-signal": "#1994ff",
      "--account-avatar-fill": "#171717",
      "--account-avatar-foreground": "#ffffff",
      "--activity-heatmap-1": "#d4d4d8",
      "--activity-heatmap-2": "#a1a1aa",
      "--activity-heatmap-3": "#71717a",
      "--activity-heatmap-4": "#3f3f46",
      "--status-success": "#047857",
    });
    expect(Object.fromEntries(darkTokenOverrides)).toMatchObject({
      "--account-avatar-fill": "#f5f5f5",
      "--account-avatar-foreground": "#111113",
      "--activity-heatmap-1": "#3f3f46",
      "--activity-heatmap-2": "#71717a",
      "--activity-heatmap-3": "#a1a1aa",
      "--activity-heatmap-4": "#d4d4d8",
      "--status-success": "#6ee7b7",
    });
    expect(darkTokenOverrides.has("--ultra-launch-signal")).toBe(false);
    expect(stylesSource).toMatch(
      /\/\*\s*Future launch-only, non-text signal; intentionally has no var\(\) consumers\.\s*\*\/\s*--ultra-launch-signal:\s*#1994ff;/
    );
    expect(stylesSource.match(/--ultra-launch-signal/g)).toHaveLength(1);
    expect(stylesSource).not.toContain("var(--ultra-launch-signal)");
    expect(stylesSource).not.toMatch(/--ultra-accent-/);
  });

  it("preserves the chart palette while identity stops borrowing chart colors", () => {
    expect(lightTokens.get("--chart-2")).toBe("oklch(0.559 0.169 41.7)");
    expect(darkTokens.get("--chart-2")).toBe("oklch(0.665 0.169 42.0)");
    expect(stylesSource.match(/--chart-2\s*:/g)).toHaveLength(2);
    const avatarRules = groupedBlockBody(
      stylesSource,
      '.app-sidebar-account-avatar [data-slot="avatar-fallback"],'
    );
    expect(avatarRules).not.toContain("var(--chart-2)");
    expect(avatarRules).toContain("background: var(--account-avatar-fill)");
    expect(avatarRules).toContain("color: var(--account-avatar-foreground)");

    const heatmapRules = [1, 2, 3, 4]
      .map((level) => blockBody(stylesSource, `.token-heatmap-cell[data-level="${level}"]`))
      .join("\n");
    expect(heatmapRules).not.toContain("var(--chart-2)");
    for (const level of [1, 2, 3, 4]) {
      expect(heatmapRules).toContain(`var(--activity-heatmap-${level})`);
    }
  });

  it("meets avatar contrast and strict heatmap luminance requirements", () => {
    expect(
      contrastRatio(
        lightTokens.get("--account-avatar-foreground") ?? "",
        lightTokens.get("--account-avatar-fill") ?? ""
      )
    ).toBeGreaterThanOrEqual(4.5);
    expect(
      contrastRatio(
        darkTokens.get("--account-avatar-foreground") ?? "",
        darkTokens.get("--account-avatar-fill") ?? ""
      )
    ).toBeGreaterThanOrEqual(4.5);

    const lightLuminance = [1, 2, 3, 4].map((level) =>
      relativeLuminance(lightTokens.get(`--activity-heatmap-${level}`) ?? "")
    );
    const darkLuminance = [1, 2, 3, 4].map((level) =>
      relativeLuminance(darkTokens.get(`--activity-heatmap-${level}`) ?? "")
    );
    expect(
      lightLuminance.every(
        (value, index) => index === 0 || value < lightLuminance[index - 1]
      )
    ).toBe(true);
    expect(
      darkLuminance.every(
        (value, index) => index === 0 || value > darkLuminance[index - 1]
      )
    ).toBe(true);
    expect(
      contrastRatio(lightTokens.get("--activity-heatmap-4") ?? "", "#ffffff")
    ).toBeGreaterThanOrEqual(3);
    expect(
      contrastRatio(darkTokens.get("--activity-heatmap-4") ?? "", "#111113")
    ).toBeGreaterThanOrEqual(3);
  });

  it("preserves success, warning, danger, and scientific color boundaries", () => {
    expect(lightTokens.get("--status-success")).toBe("#047857");
    expect(darkTokens.get("--status-success")).toBe("#6ee7b7");
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

    expect(lightTokens.get("--danger")).toBe("#c62828");
    expect(lightTokens.get("--destructive")).toBe("oklch(0.577 0.245 27.325)");
    expect(darkTokens.get("--destructive")).toBe("oklch(0.704 0.191 22.216)");
    const runtimeWarningRules = blockBody(stylesSource, ".admin-runtime-warn");
    expect(runtimeWarningRules).toContain("var(--chart-4)");
    expect(runtimeWarningRules).not.toContain("var(--status-success)");

    const scientificLightTokens = variables(blockBody(stylesSource, ":root", "--tv-string:"));
    const scientificDarkTokens = variables(blockBody(stylesSource, ".dark", "--tv-string:"));
    expect(Object.fromEntries(scientificLightTokens)).toMatchObject({
      "--tv-string": "#0f6e56",
      "--tv-number": "#185fa5",
      "--tv-bool": "#993c1d",
      "--tv-link": "#185fa5",
    });
    expect(Object.fromEntries(scientificDarkTokens)).toMatchObject({
      "--tv-string": "#5dcaa5",
      "--tv-number": "#85b7eb",
      "--tv-bool": "#f0997b",
      "--tv-link": "#85b7eb",
    });
  });

  it("preserves representative chart-2 data-series consumers", () => {
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
  });

  it("uses accessible monochrome context and emphasis in every wordmark theme", () => {
    expect(Object.fromEntries(typographyLightTokens)).toMatchObject({
      "--brand-wordmark-context-light": "#52525b",
      "--brand-wordmark-emphasis-light": "#171717",
      "--brand-wordmark-context-dark": "#a1a1aa",
      "--brand-wordmark-emphasis-dark": "#f5f5f5",
      "--brand-wordmark-context": "var(--brand-wordmark-context-light)",
      "--brand-wordmark-emphasis": "var(--brand-wordmark-emphasis-light)",
    });
    expect(typographyDarkTokens.get("--brand-wordmark-context")).toBe(
      "var(--brand-wordmark-context-dark)"
    );
    expect(typographyDarkTokens.get("--brand-wordmark-emphasis")).toBe(
      "var(--brand-wordmark-emphasis-dark)"
    );

    for (const color of ["#52525b", "#171717"]) {
      expect(contrastRatio(color, "#ffffff")).toBeGreaterThanOrEqual(4.5);
    }
    for (const color of ["#a1a1aa", "#f5f5f5"]) {
      expect(contrastRatio(color, "#111113")).toBeGreaterThanOrEqual(4.5);
    }
    expect(typographyLightTokens.get("--brand-wordmark-context-light")).not.toBe(
      typographyLightTokens.get("--brand-wordmark-emphasis-light")
    );
    expect(typographyLightTokens.get("--brand-wordmark-context-dark")).not.toBe(
      typographyLightTokens.get("--brand-wordmark-emphasis-dark")
    );

    const bisqueRules = blockBody(typographySource, ".brand-wordmark__bisque");
    expect(bisqueRules).toContain("color: var(--brand-wordmark-context)");
    expect(bisqueRules).toContain("font-weight: 400");
    const ultraRules = blockBody(typographySource, ".brand-wordmark__ultra");
    expect(ultraRules).toContain("color: var(--brand-wordmark-emphasis)");
    expect(ultraRules).toContain("font-weight: 600");

    const authRules = blockBody(typographySource, ".auth-screen-logo");
    expect(authRules).toContain(
      "--brand-wordmark-context: var(--brand-wordmark-context-dark)"
    );
    expect(authRules).toContain(
      "--brand-wordmark-emphasis: var(--brand-wordmark-emphasis-dark)"
    );
    expect(typographySource).not.toContain("--brand-wordmark-accent");
  });
});
