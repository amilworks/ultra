import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(import.meta.dirname, "..");
const distAssetsDir = path.join(repoRoot, "dist", "assets");

const budgets = [
  { label: "app-shell", pattern: /^index-.*\.js$/, maxBytes: 600_000 },
  { label: "upload-viewer", pattern: /^UploadViewerSheet-.*\.js$/, maxBytes: 120_000 },
  { label: "vendor-ui", pattern: /^vendor-ui-.*\.js$/, maxBytes: 225_000 },
  { label: "vendor-markdown", pattern: /^vendor-markdown-.*\.js$/, maxBytes: 500_000 },
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

const failures = [];

for (const budget of budgets) {
  const asset = assets.find(({ file }) => budget.pattern.test(file));
  if (!asset) {
    failures.push(
      `Missing expected bundle chunk for ${budget.label}. Pattern: ${budget.pattern}`
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
