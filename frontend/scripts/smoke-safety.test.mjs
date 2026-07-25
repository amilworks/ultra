import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import http from "node:http";
import path from "node:path";
import process from "node:process";
import test from "node:test";

import { chromium } from "playwright";

import { attachSmokePageGuard } from "./smoke-page-guard.mjs";
import {
  stopProcess,
  waitForGuardedProcess,
  waitForProcess,
} from "./smoke-process.mjs";
import { createTypographyRequestAudit } from "./typography-request-audit.mjs";

const frontendRoot = path.resolve(import.meta.dirname, "..");

const listen = (server, port = 0) =>
  new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => {
      server.off("error", reject);
      resolve(server.address().port);
    });
  });

const close = (server) => new Promise((resolve) => server.close(resolve));

// Poll a predicate until it holds or the budget runs out. Used for effects that are
// NOT covered by a page lifecycle event — e.g. a request issued by an inline script,
// which the route handler can record after `goto(..., {waitUntil:"load"})` resolves.
// Returns the predicate's final value so the caller still owns the assertion.
const waitFor = async (predicate, { timeoutMs = 5000, intervalMs = 25 } = {}) => {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (predicate()) {
      return true;
    }
    if (Date.now() >= deadline) {
      return false;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
};

const reserveFreePort = async () => {
  const server = http.createServer();
  const port = await listen(server);
  await close(server);
  return port;
};

const runNodeScript = (script, env) =>
  new Promise((resolve) => {
    const child = spawn(process.execPath, [script], {
      cwd: frontendRoot,
      env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk;
    });
    child.stderr.on("data", (chunk) => {
      output += chunk;
    });
    child.once("exit", (code, signal) => resolve({ code, signal, output }));
  });

const createCanary = async () => {
  const requests = [];
  const server = http.createServer((request, response) => {
    requests.push({ method: request.method, url: request.url });
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end('{"service":"canary"}');
  });
  const port = await listen(server);
  return { port, requests, server };
};

test("reserved live port 5174 is rejected before any canary request", async () => {
  const canary = await createCanary();
  try {
    const result = await runNodeScript("scripts/run-mobile-smoke.mjs", {
      SMOKE_API_PORT: String(canary.port),
      SMOKE_WEB_PORT: "5174",
    });
    assert.notEqual(result.code, 0);
    assert.match(result.output, /must never use reserved live API port 5174/);
    assert.deepEqual(canary.requests, []);
  } finally {
    await close(canary.server);
  }
});

test("an occupied explicit mock port fails closed without probing its listener", async () => {
  const canary = await createCanary();
  try {
    const webPort = await reserveFreePort();
    const result = await runNodeScript("scripts/run-mobile-smoke.mjs", {
      SMOKE_API_PORT: String(canary.port),
      SMOKE_WEB_PORT: String(webPort),
    });
    assert.notEqual(result.code, 0);
    assert.match(result.output, /already occupied/);
    assert.deepEqual(canary.requests, []);
  } finally {
    await close(canary.server);
  }
});

test("direct browser runner rejects a missing wrapper nonce without interaction", async () => {
  const canary = await createCanary();
  try {
    const result = await runNodeScript("scripts/mobile-smoke.mjs", {
      MOBILE_SMOKE_NONCE: "",
      MOBILE_SMOKE_URL: `http://127.0.0.1:${canary.port}`,
    });
    assert.notEqual(result.code, 0);
    assert.match(result.output, /wrapper-issued MOBILE_SMOKE_NONCE/);
    assert.deepEqual(canary.requests, []);
  } finally {
    await close(canary.server);
  }
});

test("a post-identity listener replacement cannot receive an account-request POST", async () => {
  const nonce = randomBytes(32).toString("hex");
  const replacementRequests = [];
  let replacementServer;
  let replacementListeningResolve;
  const replacementListening = new Promise((resolve) => {
    replacementListeningResolve = resolve;
  });
  const identityServer = http.createServer((request, response) => {
    assert.equal(request.url, "/v1/smoke/identity");
    response.writeHead(200, {
      "Connection": "close",
      "Content-Type": "application/json",
    });
    response.end(
      JSON.stringify({
        service: "ultra-mobile-smoke-mock",
        nonce,
      })
    );
    response.once("finish", () => {
      identityServer.close(() => {
        replacementServer = http.createServer((replacementRequest, replacementResponse) => {
          replacementRequests.push({
            method: replacementRequest.method,
            url: replacementRequest.url,
          });
          replacementResponse.writeHead(200, { "Content-Type": "text/html" });
          if (replacementRequest.method === "POST") {
            replacementResponse.end(
              '<div class="app-shell"><textarea class="pk-prompt-input-textarea"></textarea></div>'
            );
            return;
          }
          replacementResponse.end(`
            <div class="auth-screen">
              <input id="guest-name">
              <input id="guest-email">
              <input id="guest-affiliation">
              <form class="auth-form" method="POST" action="/v2/auth/request-account">
                <button type="submit">Request an Account</button>
              </form>
            </div>
          `);
        });
        replacementServer.listen(port, "127.0.0.1", () => {
          replacementListeningResolve();
        });
      });
    });
  });
  const port = await listen(identityServer);

  try {
    const resultPromise = runNodeScript("scripts/mobile-smoke.mjs", {
      MOBILE_SMOKE_NONCE: nonce,
      MOBILE_SMOKE_URL: `http://127.0.0.1:${port}`,
    });
    await replacementListening;
    const result = await resultPromise;
    assert.notEqual(result.code, 0);
    assert.ok(
      replacementRequests.some(({ method }) => method === "GET"),
      "replacement listener was never exercised"
    );
    assert.deepEqual(
      replacementRequests.filter(({ method }) => method === "POST"),
      []
    );
  } finally {
    if (identityServer.listening) {
      await close(identityServer);
    }
    if (replacementServer?.listening) {
      await close(replacementServer);
    }
  }
});

test("actual Playwright wiring records and rejects an attempted foreign stylesheet", async () => {
  const foreignRequests = [];
  const localRequests = [];
  const foreignServer = http.createServer((request, response) => {
    foreignRequests.push({ method: request.method, url: request.url });
    response.writeHead(200, { "Content-Type": "text/css" });
    response.end("body { color: red; }");
  });
  const foreignPort = await listen(foreignServer);
  let localPort;
  const localServer = http.createServer((request, response) => {
    localRequests.push({ method: request.method, url: request.url });
    if (request.url === "/local.css") {
      response.writeHead(200, { "Content-Type": "text/css" });
      response.end("body { color: black; }");
      return;
    }
    response.writeHead(200, { "Content-Type": "text/html" });
    response.end(`
      <link rel="stylesheet" href="/local.css">
      <link rel="stylesheet" href="http://127.0.0.1:${foreignPort}/foreign.css">
      <script>fetch("/v2/auth/request-account", { method: "POST" }).catch(() => {});</script>
      <main>adapter canary</main>
    `);
  });
  localPort = await listen(localServer);
  const baseUrl = `http://127.0.0.1:${localPort}`;
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    const audit = createTypographyRequestAudit(baseUrl);
    const guard = await attachSmokePageGuard(page, {
      baseUrl,
      typographyAudit: audit,
    });
    await page.goto(baseUrl, { waitUntil: "load" });

    assert.deepEqual(foreignRequests, []);
    assert.ok(
      guard.blockedRequests.some(({ url }) => url.endsWith("/foreign.css")),
      "foreign stylesheet was not intercepted by the page guard"
    );
    // The two stylesheet links are fetched while the document parses, so `load`
    // already guarantees the guard saw them. The account-request POST is issued by an
    // INLINE SCRIPT, which `load` does not wait on — the guard can record it after
    // goto() resolves, so asserting immediately races it. Wait for the interception
    // (bounded) rather than for a lucky ordering; the assertion itself is unchanged.
    const interceptedAccountRequest = await waitFor(() =>
      guard.blockedRequests.some(
        ({ method, url }) =>
          method === "POST" && url.endsWith("/v2/auth/request-account")
      )
    );
    assert.ok(
      interceptedAccountRequest,
      "account-request mutation was not intercepted by the page guard"
    );
    assert.deepEqual(
      localRequests.filter(({ method }) => method === "POST"),
      []
    );
    assert.throws(
      () => audit.assertLocalSuccess(assert.ok, "adapter-canary"),
      /attempted remote font\/CSS request/
    );
    await guard.detach();
  } finally {
    await browser.close();
    await Promise.all([close(localServer), close(foreignServer)]);
  }
});

test("smoke timeout awaits TERM grace and KILL cleanup before rejecting", async () => {
  const child = spawn(
    process.execPath,
    [
      "-e",
      "process.on('SIGTERM',()=>{}); console.log('ready'); setInterval(()=>{},1000)",
    ],
    { stdio: ["ignore", "pipe", "ignore"] }
  );
  await new Promise((resolve) => child.stdout.once("data", resolve));
  const startedAt = Date.now();
  await assert.rejects(
    waitForProcess(child, {
      label: "Cleanup fixture",
      timeoutMs: 40,
      termGraceMs: 40,
    }),
    /exceeded 40ms/
  );
  assert.equal(child.signalCode, "SIGKILL");
  assert.ok(Date.now() - startedAt >= 75, "timeout rejected before cleanup grace elapsed");
  assert.ok(Date.now() - startedAt < 2_000, "timeout cleanup exceeded its bounded window");
  await stopProcess(child);
});

test("authority exit terminates the browser worker before failing", async () => {
  const worker = spawn(
    process.execPath,
    [
      "-e",
      "process.on('SIGTERM',()=>{}); console.log('ready'); setInterval(()=>{},1000)",
    ],
    { stdio: ["ignore", "pipe", "ignore"] }
  );
  await new Promise((resolve) => worker.stdout.once("data", resolve));
  const authority = spawn(
    process.execPath,
    ["-e", "setTimeout(()=>process.exit(0),25)"],
    { stdio: "ignore" }
  );

  await assert.rejects(
    waitForGuardedProcess(worker, {
      authorities: [{ child: authority, label: "Canary authority" }],
      label: "Canary browser",
      timeoutMs: 2_000,
      termGraceMs: 40,
    }),
    /Canary authority exited unexpectedly/
  );
  assert.equal(worker.signalCode, "SIGKILL");
  await Promise.all([stopProcess(worker), stopProcess(authority)]);
});
