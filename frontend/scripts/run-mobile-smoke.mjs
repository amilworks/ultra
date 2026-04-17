import { spawn } from "node:child_process";
import net from "node:net";
import path from "node:path";
import process from "node:process";

const repoRoot = path.resolve(process.cwd());
const frontendRoot = repoRoot;
const mockPort = Number(process.env.SMOKE_API_PORT || "18000");
const vitePort = Number(process.env.SMOKE_WEB_PORT || "15173");

const startProcess = (command, args, env = {}) =>
  spawn(command, args, {
    cwd: frontendRoot,
    env: {
      ...process.env,
      ...env,
    },
    stdio: "inherit",
  });

const waitForPort = (port, timeoutMs = 30000) =>
  new Promise((resolve, reject) => {
    const startedAt = Date.now();
    const tryConnect = () => {
      const socket = net.createConnection({ host: "127.0.0.1", port }, () => {
        socket.end();
        resolve();
      });
      socket.on("error", () => {
        socket.destroy();
        if (Date.now() - startedAt >= timeoutMs) {
          reject(new Error(`Timed out waiting for port ${port}`));
          return;
        }
        setTimeout(tryConnect, 250);
      });
    };
    tryConnect();
  });

const stopProcess = (child) =>
  new Promise((resolve) => {
    if (!child || child.exitCode !== null) {
      resolve();
      return;
    }
    child.once("exit", () => resolve());
    child.kill("SIGTERM");
    setTimeout(() => {
      if (child.exitCode === null) {
        child.kill("SIGKILL");
      }
    }, 2000);
  });

const mockApi = startProcess("node", ["scripts/mock-api.mjs"], {
  MOCK_API_PORT: String(mockPort),
});
const vite = startProcess("pnpm", ["exec", "vite", "--host", "127.0.0.1", "--port", String(vitePort)], {
  VITE_PROXY_API_TARGET: `http://127.0.0.1:${mockPort}`,
});

try {
  await waitForPort(mockPort);
  await waitForPort(vitePort);

  await new Promise((resolve, reject) => {
    const smoke = startProcess("node", ["scripts/mobile-smoke.mjs"], {
      MOBILE_SMOKE_URL: `http://127.0.0.1:${vitePort}`,
    });
    smoke.once("exit", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`Mobile smoke exited with status ${code}`));
    });
  });
} finally {
  await Promise.all([stopProcess(vite), stopProcess(mockApi)]);
}
