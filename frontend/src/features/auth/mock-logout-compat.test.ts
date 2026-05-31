/// <reference types="node" />

import { spawn, type ChildProcess } from "node:child_process";
import net from "node:net";

import { afterEach, describe, expect, it } from "vitest";

let child: ChildProcess | null = null;

const getFreePort = (): Promise<number> =>
  new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close(() => reject(new Error("Unable to allocate a test port.")));
        return;
      }
      const { port } = address;
      server.close(() => resolve(port));
    });
  });

const waitForPort = (port: number): Promise<void> =>
  new Promise((resolve, reject) => {
    const startedAt = Date.now();
    const tryConnect = () => {
      const socket = net.createConnection({ host: "127.0.0.1", port }, () => {
        socket.end();
        resolve();
      });
      socket.once("error", () => {
        socket.destroy();
        if (Date.now() - startedAt > 5000) {
          reject(new Error(`Timed out waiting for mock API port ${port}`));
          return;
        }
        setTimeout(tryConnect, 50);
      });
    };
    tryConnect();
  });

afterEach(async () => {
  const activeChild = child;
  child = null;
  if (!activeChild || activeChild.exitCode !== null) {
    return;
  }
  await new Promise<void>((resolve) => {
    activeChild.once("exit", () => resolve());
    activeChild.kill("SIGTERM");
    setTimeout(() => {
      if (activeChild.exitCode === null) {
        activeChild.kill("SIGKILL");
      }
    }, 1000);
  });
});

describe("mock API logout compatibility", () => {
  it("redirects stale browser logout links back to the app", async () => {
    const port = await getFreePort();
    child = spawn("node", ["scripts/mock-api.mjs"], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        MOCK_API_PORT: String(port),
      },
      stdio: "ignore",
    });
    await waitForPort(port);

    const nextUrl = "http://localhost:5174/?conversation=thread_575";
    const response = await fetch(
      `http://127.0.0.1:${port}/v1/auth/logout/browser?next=${encodeURIComponent(nextUrl)}`,
      { redirect: "manual" }
    );

    expect(response.status).toBe(302);
    expect(response.headers.get("location")).toBe(nextUrl);
    expect(response.headers.get("set-cookie")).toContain("Max-Age=0");
  });
});
