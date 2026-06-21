/// <reference types="node" />

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const readConfig = (filename: string) =>
  readFileSync(path.join(process.cwd(), filename), "utf8");

const expectVite8Chunking = (source: string) => {
  expect(source).toContain("rolldownOptions");
  expect(source).toContain("@radix-ui");
  expect(source).toContain("@floating-ui");
  expect(source).toContain("tailwind-merge");
  expect(source).not.toContain("rollupOptions");
};

describe("frontend build chunking", () => {
  it("keeps Vite 8 UI dependencies out of the committed app shell config", () => {
    expectVite8Chunking(readConfig("vite.config.ts"));
  });

  it("keeps the Lens route benchmark selector aligned with the sidebar label", () => {
    const source = readConfig("scripts/ui-performance-benchmark.mjs");

    expect(source).toContain('button[title^="Lens"]');
    expect(source).not.toContain('button[title="Scientific Viewer"]');
  });

  it("keeps z-scrub thumbnails from statically pulling the Three-backed slice cache", () => {
    const source = readConfig("src/components/viewer/ZScrubThumbnail.tsx");

    expect(source).not.toContain('from "./sliceImageCache"');
    expect(source).toContain('import("./sliceImageCache")');
  });

  it("keeps the ignored local Vite JS mirror in sync when present", () => {
    const localConfigPath = path.join(process.cwd(), "vite.config.js");
    if (!existsSync(localConfigPath)) {
      return;
    }

    expectVite8Chunking(readConfig("vite.config.js"));
  });
});
