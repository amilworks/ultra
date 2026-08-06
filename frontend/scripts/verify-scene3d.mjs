/**
 * End-to-end verification of the scene3d wire format on a real GPU.
 *
 * Unit tests prove the encoder matches Spark's byte-for-byte and that the pure modules
 * are correct, but neither can prove the two halves meet: that the bytes the Python
 * derive writes are what Spark's renderer actually consumes. This drives the built
 * harness with Playwright and asserts on real rendered state.
 *
 * Deliberately self-contained — it starts an ephemeral static server inside this process
 * and tears it down, rather than leaving a dev server running.
 *
 *   node scripts/verify-scene3d.mjs --fixtures <dir> [--out <dir>] [--scene splats,points]
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { chromium } from "playwright";

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i].replace(/^--/, ""), process.argv[i + 1]);
}

const harnessDir = args.get("harness") ?? path.resolve(import.meta.dirname, "../../.tmp/scene3d-harness");
const fixturesDir = args.get("fixtures");
const outDir = args.get("out") ?? path.resolve(import.meta.dirname, "../../.tmp/scene3d-shots");
const scenes = (args.get("scene") ?? "splats,points").split(",");
const budget = args.get("budget") ?? "80";

if (!fixturesDir || !fs.existsSync(fixturesDir)) {
  throw new Error(`--fixtures <dir> is required and must exist (got ${fixturesDir})`);
}
if (!fs.existsSync(harnessDir)) {
  throw new Error(`harness build not found at ${harnessDir}; run vite build -c vite.harness.config.ts first`);
}
fs.mkdirSync(outDir, { recursive: true });

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".bin": "application/octet-stream",
  ".png": "image/png",
  ".css": "text/css; charset=utf-8",
};

// Serve the harness build at /, and the derived fixture at /scene3d-fixture/.
const server = http.createServer((req, res) => {
  const url = decodeURIComponent((req.url ?? "/").split("?")[0]);
  const [root, rel] = url.startsWith("/scene3d-fixture/")
    ? [fixturesDir, url.slice("/scene3d-fixture/".length)]
    : [harnessDir, url === "/" ? "scene3d-harness.html" : url.slice(1)];
  const file = path.resolve(root, rel);
  if (!file.startsWith(path.resolve(root))) {
    res.writeHead(403).end("forbidden");
    return;
  }
  fs.readFile(file, (err, body) => {
    if (err) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
    res.end(body);
  });
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;

const browser = await chromium.launch({
  args: ["--use-gl=angle", "--use-angle=metal", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const results = [];
let failed = false;

for (const scene of scenes) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const consoleErrors = [];
  page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
  page.on("pageerror", (e) => consoleErrors.push(String(e)));

  const url = `http://127.0.0.1:${port}/scene3d-harness.html?scene=${scene}&budget=${budget}&tier=0`;
  await page.goto(url, { waitUntil: "load" });

  let state;
  try {
    await page.waitForFunction(() => window.harness?.ready || window.harness?.error, { timeout: 180_000 });
    state = await page.evaluate(() => window.harness);
  } catch (error) {
    state = { ready: false, error: `timed out: ${error.message}` };
  }

  const shot = path.join(outDir, `scene3d-${scene}.png`);
  await page.screenshot({ path: shot });

  const ok = Boolean(state.ready) && !state.error && state.nonBlack > 0.02;
  if (!ok) failed = true;
  results.push({ scene, ok, shot, consoleErrors, ...state });

  console.log(`\n=== ${scene} ===`);
  if (state.error) console.log(`  ERROR         ${state.error}`);
  else {
    console.log(`  scene_kind    ${state.sceneKind}`);
    console.log(`  chunks        ${state.chunksLoaded}/${state.chunksTotal}  (${(state.bytes / 1e6).toFixed(1)} MB)`);
    console.log(`  elements      ${state.elementsLoaded?.toLocaleString("en-US")} of ${state.elementsTotal?.toLocaleString("en-US")}`);
    console.log(`  spark active  ${state.hasSpark}`);
    console.log(`  lit pixels    ${(state.nonBlack * 100).toFixed(1)}%  ${state.nonBlack > 0.02 ? "OK" : "TOO DARK / EMPTY"}`);
  }
  if (consoleErrors.length) console.log(`  console errors:\n    ${consoleErrors.slice(0, 5).join("\n    ")}`);
  console.log(`  screenshot    ${shot}`);
  await page.close();
}

await browser.close();
server.close();

fs.writeFileSync(path.join(outDir, "results.json"), JSON.stringify(results, null, 2));
console.log(`\n${failed ? "FAILED" : "PASSED"} — results in ${outDir}/results.json`);
process.exit(failed ? 1 : 0);
