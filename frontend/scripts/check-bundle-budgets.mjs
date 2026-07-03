import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(import.meta.dirname, "..");
const distAssetsDir = path.join(repoRoot, "dist", "assets");
const manifestPath = path.join(repoRoot, "dist", ".vite", "manifest.json");

const budgets = [
  { label: "app-shell", pattern: /^index-.*\.js$/, maxBytes: 490_000 },
  { label: "auth-screen", pattern: /^AuthScreen-.*\.js$/, maxBytes: 25_000 },
  { label: "composer-slash-menu", pattern: /^ComposerSlashMenu-.*\.js$/, maxBytes: 20_000 },
  { label: "composer-workflows", pattern: /^composer-workflows-.*\.js$/, maxBytes: 20_000 },
  { label: "chat-run-steps", pattern: /^ChatRunSteps-.*\.js$/, maxBytes: 15_000 },
  {
    label: "inline-data-quick-preview",
    pattern: /^InlineDataQuickPreview-.*\.js$/,
    maxBytes: 20_000,
  },
  {
    label: "tool-result-quick-preview",
    pattern: /^ToolResultQuickPreview-.*\.js$/,
    maxBytes: 12_000,
  },
  { label: "tool-result-cards", pattern: /^ToolResultCards-.*\.js$/, maxBytes: 32_000 },
  { label: "tool-image-carousel", pattern: /^ToolImageCarousel-.*\.js$/, maxBytes: 16_000 },
  { label: "upload-viewer", pattern: /^UploadViewerSheet-.*\.js$/, maxBytes: 120_000 },
  { label: "vendor-ui", pattern: /^vendor-ui-.*\.js$/, maxBytes: 250_000 },
  { label: "route-charts", pattern: /^chart-.*\.js$/, maxBytes: 380_000 },
  { label: "viewer-manifest", pattern: /^viewerManifest-.*\.js$/, maxBytes: 60_000 },
  {
    label: "prompt-markdown",
    manifestKey: "src/components/prompt-kit/markdown.tsx",
    maxBytes: 220_000,
  },
  { label: "rehype-katex", pattern: /^rehype-katex-.*\.js$/, maxBytes: 300_000 },
  { label: "shiki-runtime", pattern: /^shiki-.*\.js$/, maxBytes: 165_000 },
  { label: "vendor-three", pattern: /^vendor-three-.*\.js$/, maxBytes: 560_000 },
];

const formatBytes = (value) => `${(value / 1024).toFixed(1)} KiB`;

if (!fs.existsSync(distAssetsDir)) {
  throw new Error(`Bundle assets directory not found: ${distAssetsDir}`);
}

const assets = fs
  .readdirSync(distAssetsDir)
  .filter((entry) => entry.endsWith(".js"))
  .map((entry) => {
    const fullPath = path.join(distAssetsDir, entry);
    return {
      file: entry,
      size: fs.statSync(fullPath).size,
    };
  });

const manifest = fs.existsSync(manifestPath)
  ? JSON.parse(fs.readFileSync(manifestPath, "utf8"))
  : {};
const failures = [];

const resolveBudgetAsset = (budget) => {
  if (budget.manifestKey) {
    const manifestEntry = manifest[budget.manifestKey];
    const file = manifestEntry?.file ? path.basename(manifestEntry.file) : null;
    return file ? assets.find((asset) => asset.file === file) : undefined;
  }
  const matches = assets.filter(({ file }) => budget.pattern.test(file));
  return matches.toSorted((left, right) => right.size - left.size)[0];
};

for (const budget of budgets) {
  const asset = resolveBudgetAsset(budget);
  if (!asset) {
    failures.push(
      budget.manifestKey
        ? `Missing expected bundle chunk for ${budget.label}. Manifest key: ${budget.manifestKey}`
        : `Missing expected bundle chunk for ${budget.label}. Pattern: ${budget.pattern}`
    );
    continue;
  }
  if (asset.size > budget.maxBytes) {
    failures.push(
      `${budget.label} exceeded budget: ${asset.file} is ${formatBytes(asset.size)} (budget ${formatBytes(
        budget.maxBytes
      )})`
    );
  }
}

const sortedAssets = [...assets].sort((left, right) => right.size - left.size);
console.log("Frontend bundle report:");
for (const asset of sortedAssets) {
  console.log(`- ${asset.file}: ${formatBytes(asset.size)}`);
}

if (failures.length > 0) {
  throw new Error(failures.join("\n"));
}
