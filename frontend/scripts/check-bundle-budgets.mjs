import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(import.meta.dirname, "..");
const distAssetsDir = path.join(repoRoot, "dist", "assets");
const manifestPath = path.join(repoRoot, "dist", ".vite", "manifest.json");

const budgets = [
  // 2026-09-02: the composer brief (inline @tokens, prefix chips, whisper) is
  // first-paint UI and lives in the shell; the @ picker is its own chunk below.
  // Shell measured 475.1 KiB before it and 488.4 KiB after (+13.3 KiB):
  // budget 490_000 → 505_000, keeping the same ~5 KiB of headroom.
  { label: "app-shell", pattern: /^index-.*\.js$/, maxBytes: 505_000 },
  // The composer's editor is its own chunk; ProseMirror core is shared with Notes.
  { label: "composer-editor", pattern: /^ComposerEditor-.*\.js$/, maxBytes: 30_000 },
  { label: "vendor-prosemirror", pattern: /^vendor-prosemirror-.*\.js$/, maxBytes: 420_000 },
  { label: "auth-screen", pattern: /^AuthScreen-.*\.js$/, maxBytes: 25_000 },
  { label: "composer-slash-menu", pattern: /^ComposerSlashMenu-.*\.js$/, maxBytes: 20_000 },
  { label: "composer-workflows", pattern: /^composer-workflows-.*\.js$/, maxBytes: 20_000 },
  { label: "notes-access", pattern: /^notes-access-.*\.js$/, maxBytes: 18_000 },
  { label: "notes-recovery", pattern: /^notes-recovery-.*\.js$/, maxBytes: 18_000 },
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
  // The Gaussian-splat runtime. Big, but lazy: only the scene3d viewer imports it,
  // it is excluded from modulePreload, and it never enters the app shell.
  { label: "vendor-spark", pattern: /^vendor-spark-.*\.js$/, maxBytes: 5_600_000 },
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

// Chunk-isolation invariants. A size budget alone cannot catch these: three.js is
// imported by both the volume viewer and Spark's splat runtime, and when Rolldown
// hoists the shared three core into vendor-spark, vendor-three shrinks to a stub that
// imports it — every budget still passes while opening a NIfTI volume silently pulls
// 5 MB of Gaussian-splat runtime. Assert the dependency direction, not just the bytes.
const isolationRules = [
  {
    from: /^index-.*\.js$/,
    mustNotImport: "vendor-prosemirror",
    why: "the app shell must not pull the editor runtime; the composer loads it lazily",
  },
  {
    from: /^SliceStackVolumeCanvas-.*\.js$/,
    mustNotImport: "vendor-spark",
    why: "the volume viewer must not pull the Gaussian-splat runtime",
  },
  {
    from: /^vendor-three-.*\.js$/,
    mustNotImport: "vendor-spark",
    why: "three.js core must live in vendor-three, not inside vendor-spark",
  },
];

for (const rule of isolationRules) {
  const match = assets.find((asset) => rule.from.test(asset.file));
  if (!match) {
    failures.push(`Missing expected bundle chunk for isolation rule. Pattern: ${rule.from}`);
    continue;
  }
  const source = fs.readFileSync(path.join(distAssetsDir, match.file), "utf8");
  // A STATIC import is the violation. Vite also lists a lazy chunk's own
  // dependencies as preload strings in whoever imports it dynamically — that
  // string names the chunk without loading it, and is not a dependency edge.
  const staticImport = new RegExp(`(?:from|import)\\s*["']\\./${rule.mustNotImport}`);
  if (staticImport.test(source)) {
    failures.push(
      `${match.file} must not import ${rule.mustNotImport}: ${rule.why}. ` +
        `Check the advancedChunks group priorities in vite.config.ts.`
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
