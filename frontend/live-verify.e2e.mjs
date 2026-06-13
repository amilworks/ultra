// Live browser verification of the control-plane changes (ingest fast path,
// thread-messages index, NATS resilience, admin authz). Temporary harness —
// run from frontend/ so playwright resolves: node live-verify.e2e.mjs [tag]
import { chromium } from "playwright";
import { readFileSync, mkdirSync, appendFileSync } from "node:fs";

const TAG = process.argv[2] || "run";
const OUT = "../.tmp/e2e/live-verify";
mkdirSync(OUT, { recursive: true });
const COOKIE = readFileSync("../.tmp/e2e/session-cookie.txt", "utf8").trim();
const log = (line) => {
  console.log(line);
  appendFileSync(`${OUT}/results-${TAG}.log`, line + "\n");
};

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
await context.addCookies([
  { name: "ultra_workos_session", value: COOKIE, domain: "localhost", path: "/" },
  { name: "ultra_workos_session", value: COOKIE, domain: "127.0.0.1", path: "/" },
]);
const page = await context.newPage();

const consoleErrors = [];
page.on("console", (msg) => {
  if (msg.type() === "error") consoleErrors.push(msg.text().slice(0, 200));
});
const failedRequests = [];
let runId = "";
let sseStatus = 0;
page.on("response", async (res) => {
  const url = res.url();
  if (res.status() >= 500) failedRequests.push(`${res.status()} ${url.slice(0, 120)}`);
  if (res.status() === 404) log(`404: ${url.slice(0, 140)}`);
  if (url.includes("/runs") && res.request().method() === "POST" && res.status() < 300) {
    try {
      const body = await res.json();
      if (body && body.run_id) runId = body.run_id;
    } catch {}
  }
  if (url.includes("/events") && url.includes("stream")) sseStatus = res.status();
});

let failures = 0;
const check = (ok, label) => {
  log(`${ok ? "PASS" : "FAIL"}  ${label}`);
  if (!ok) failures += 1;
};

// --- Flow 1: authenticated app load (cold vite dev compile can be slow) ---
await page.goto("http://localhost:5174/", { waitUntil: "domcontentloaded" });
const composer = page.locator('textarea[placeholder="Ask anything"]');
await composer.first().waitFor({ state: "visible", timeout: 90000 }).catch((err) => log(`composer waitFor: ${String(err).slice(0, 120)}`));
const composerCount = await composer.count();
check(composerCount >= 1 && (await composer.first().isVisible()), `app loads authenticated with composer ready (matches=${composerCount})`);
await page.locator("text=New chat").first().click().catch(() => {});
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}/${TAG}-01-app-loaded.png` });

// --- Flow 2: chat run with live SSE streaming ---
const marker = `verify:${TAG}`;
const prompt = `Reply with exactly one short sentence confirming you received this. (${marker})`;
await composer.fill(prompt);
await composer.press("Enter");
log(`sent prompt at ${new Date().toISOString()}`);

// The assistant response renders after the user bubble; track the text that
// follows the unique marker and wait for it to appear and settle.
const textAfterMarker = async () => {
  const body = (await page.locator("body").innerText().catch(() => "")) || "";
  const index = body.lastIndexOf(marker);
  if (index < 0) return "";
  return body
    .slice(index + marker.length)
    .replace(/^[\s)]+/, "")
    .replace(/Ask anything[\s\S]*$/, "")
    .trim();
};
let lastText = "";
let lastChange = Date.now();
let streamingShot = false;
const deadline = Date.now() + 150000;
while (Date.now() < deadline) {
  await page.waitForTimeout(1000);
  const text = await textAfterMarker();
  if (text !== lastText) {
    lastText = text;
    lastChange = Date.now();
    if (!streamingShot && text.length > 0) {
      streamingShot = true;
      await page.screenshot({ path: `${OUT}/${TAG}-02-streaming.png` });
    }
  }
  if (lastText.length > 0 && Date.now() - lastChange > 8000) break;
}
await page.screenshot({ path: `${OUT}/${TAG}-03-response-complete.png` });
check(lastText.length > 0, `assistant response rendered (${lastText.length} chars): "${lastText.slice(0, 80).replace(/\n/g, " ")}"`);
check(sseStatus === 200 || sseStatus === 0, `event stream responded (status ${sseStatus || "proxied/no-capture"})`);
log(`run_id=${runId}`);

// --- Flow 3: reload, conversation history persists (messages index path) ---
const responseSnippet = lastText.slice(0, 40);
await page.reload({ waitUntil: "domcontentloaded" });
await composer.waitFor({ state: "visible", timeout: 60000 }).catch(() => {});
await page.waitForTimeout(4000);
const bodyText = (await page.locator("body").innerText().catch(() => "")) || "";
await page.screenshot({ path: `${OUT}/${TAG}-04-after-reload.png` });
check(bodyText.includes(marker), "user message persisted and rendered after reload");
check(responseSnippet.length > 0 && bodyText.includes(responseSnippet), "assistant response persisted after reload");

// Console errors snapshot BEFORE the deliberate 403 probe below.
const realConsoleErrors = consoleErrors.filter((e) => !e.includes("favicon"));
check(realConsoleErrors.length === 0, `no console errors${realConsoleErrors.length ? ": " + realConsoleErrors.slice(0, 3).join(" | ") : ""}`);
check(failedRequests.length === 0, `no 5xx responses${failedRequests.length ? ": " + failedRequests.slice(0, 3).join("; ") : ""}`);

// --- Flow 4: admin endpoint authorization (researcher must be denied) ---
const adminProbe = await page.evaluate(async () => {
  const res = await fetch("/v2/admin/overview", { credentials: "include" });
  return { status: res.status, body: (await res.text()).slice(0, 80) };
});
check(adminProbe.status === 403, `admin overview denied for researcher (status ${adminProbe.status}: ${adminProbe.body.trim()})`);

log(`SUMMARY ${TAG}: ${failures === 0 ? "ALL PASS" : failures + " FAILURES"}`);
await browser.close();
process.exit(failures === 0 ? 0 : 1);
