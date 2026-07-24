import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import net from "node:net";
import path from "node:path";
import process from "node:process";

import { stopProcess, waitForGuardedProcess } from "./smoke-process.mjs";

const frontendRoot = path.resolve(process.cwd());
const smokeTimeoutMs = 150_000;
const forbiddenPort = 5_174;
const nonce = randomBytes(32).toString("hex");

const parsePort = (name, fallback) => {
  const raw = process.env[name];
  const port = Number(raw || fallback);
  if (!Number.isInteger(port) || port < 1_024 || port > 65_535) {
    throw new Error(`${name} must be an integer from 1024 through 65535`);
  }
  if (port === forbiddenPort) {
    throw new Error(`${name} must never use reserved live API port 5174`);
  }
  return { explicit: Boolean(raw), port };
};

const mockPreference = parsePort("SMOKE_API_PORT", 18_000);
const vitePreference = parsePort("SMOKE_WEB_PORT", 15_173);

const startProcess = (command, args, env = {}) =>
  spawn(command, args, {
    cwd: frontendRoot,
    env: {
      ...process.env,
      ...env,
    },
    stdio: "inherit",
  });

const isPortAvailable = (port) =>
  new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => server.close(() => resolve(true)));
    server.listen(port, "127.0.0.1");
  });

const selectPort = async ({ explicit, port }, excludedPort) => {
  if (explicit) {
    if (port === excludedPort) {
      throw new Error("SMOKE_API_PORT and SMOKE_WEB_PORT must be different");
    }
    if (!(await isPortAvailable(port))) {
      throw new Error(`Explicit smoke port ${port} is already occupied`);
    }
    return port;
  }
  for (let candidate = port; candidate < port + 100; candidate += 1) {
    if (
      candidate !== forbiddenPort &&
      candidate !== excludedPort &&
      (await isPortAvailable(candidate))
    ) {
      return candidate;
    }
  }
  throw new Error(`No available smoke port found near ${port}`);
};

const fetchIdentity = async (url, timeoutMs = 30_000) => {
  const startedAt = Date.now();
  let lastError;
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(url, {
        headers: { Accept: "application/json" },
        redirect: "error",
        signal: AbortSignal.timeout(1_000),
      });
      if (response.ok) {
        const identity = await response.json();
        if (identity.service === "ultra-mobile-smoke-mock" && identity.nonce === nonce) {
          return;
        }
        throw new Error("listener returned the wrong smoke identity");
      }
      lastError = new Error(`identity endpoint returned HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out verifying ${url}: ${lastError?.message || "no response"}`);
};

const mockPort = await selectPort(mockPreference);
const vitePort = await selectPort(vitePreference, mockPort);

const mockApi = startProcess("node", ["scripts/mock-api.mjs"], {
  MOCK_API_PORT: String(mockPort),
  SMOKE_RUN_NONCE: nonce,
});
let vite;
let smoke;

try {
  // A free-port probe is inherently racy. The nonce endpoint proves that the
  // listener which actually acquired the port is this wrapper's mock process.
  await fetchIdentity(`http://127.0.0.1:${mockPort}/v1/smoke/identity`);

  vite = startProcess(
    "pnpm",
    [
      "exec",
      "vite",
      "--host",
      "127.0.0.1",
      "--port",
      String(vitePort),
      "--strictPort",
    ],
    {
      VITE_PROXY_API_TARGET: `http://127.0.0.1:${mockPort}`,
    }
  );
  // Verifying through Vite proves both exclusive web-port ownership and the
  // intended proxy target before the browser is allowed to interact.
  await fetchIdentity(`http://127.0.0.1:${vitePort}/v1/smoke/identity`);

  smoke = startProcess("node", ["scripts/mobile-smoke.mjs"], {
    MOBILE_SMOKE_NONCE: nonce,
    MOBILE_SMOKE_URL: `http://127.0.0.1:${vitePort}`,
  });
  await waitForGuardedProcess(smoke, {
    authorities: [
      { child: mockApi, label: "Smoke mock authority" },
      { child: vite, label: "Smoke Vite authority" },
    ],
    label: "Mobile smoke",
    timeoutMs: smokeTimeoutMs,
  });
} finally {
  await Promise.all([stopProcess(smoke), stopProcess(vite), stopProcess(mockApi)]);
}
