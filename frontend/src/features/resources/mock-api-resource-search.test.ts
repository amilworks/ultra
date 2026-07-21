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

const startMockApi = async (): Promise<number> => {
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
  return port;
};

type ResourceListBody = {
  resources?: Array<{ original_name?: string }>;
};

const resourceNamesForQuery = async (port: number, query: string): Promise<string[]> => {
  const params = new URLSearchParams({ limit: "10", offset: "0", q: query });
  const response = await fetch(`http://127.0.0.1:${port}/v2/resources?${params}`);
  expect(response.status).toBe(200);
  const body = (await response.json()) as ResourceListBody;
  return (body.resources || []).map((resource) => String(resource.original_name || "")).sort();
};

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

describe("mock API resource search", () => {
  it("matches numeric age predicates used by the Resources search bar", async () => {
    const port = await startMockApi();

    await expect(resourceNamesForQuery(port, "age > 60")).resolves.toEqual([
      "subject-a-nph-under70.nii.gz",
      "subject-b-nph-under70.nii.gz",
    ]);
  });

  it("matches simple NIfTI file patterns used by the Resources search bar", async () => {
    const port = await startMockApi();

    await expect(resourceNamesForQuery(port, "*.nii")).resolves.toEqual([
      "parcels_Glasser.pconn.nii",
      "rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii",
      "subject-a-nph-under70.nii.gz",
      "subject-b-nph-under70.nii.gz",
    ]);
  });
});
