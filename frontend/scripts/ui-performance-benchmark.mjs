import { spawn } from "node:child_process";
import net from "node:net";
import path from "node:path";
import { performance } from "node:perf_hooks";
import process from "node:process";

import { chromium } from "playwright";

const frontendRoot = path.resolve(process.cwd());
const preferredMockPort = Number(process.env.BENCH_API_PORT || "18100");
const preferredVitePort = Number(process.env.BENCH_WEB_PORT || "15180");
const sampleCount = Math.max(1, Number(process.env.BENCH_SAMPLES || "5"));
const settleMs = Math.max(0, Number(process.env.BENCH_SETTLE_MS || "750"));
const benchMode = process.env.BENCH_MODE === "preview" ? "preview" : "dev";
const enforceBudgets = process.env.BENCH_ENFORCE === "1";

const benchmarkBudgets = {
  preview: {
    "app-ready": { warmMedianMs: 150, p95Ms: 250 },
    "resources-route": { warmMedianMs: 120, p95Ms: 180 },
    "training-route": { warmMedianMs: 120, p95Ms: 180 },
    "scientific-viewer-route": { warmMedianMs: 110, p95Ms: 170 },
  },
};

const cases = [
  {
    name: "resources-route",
    buttonSelector: '.app-sidebar-static button[title^="Resources"]',
    readySelector: ".resource-browser",
  },
  {
    name: "training-route",
    buttonSelector: '.app-sidebar-static button[title^="Training dashboard"]',
    readySelector: "text=Model Health",
  },
  {
    name: "scientific-viewer-route",
    buttonSelector: '.app-sidebar-static button[title="Scientific Viewer"]',
    readySelector: ".viewer-workspace",
  },
];

const startProcess = (command, args, env = {}) =>
  spawn(command, args, {
    cwd: frontendRoot,
    env: {
      ...process.env,
      ...env,
    },
    stdio: process.env.BENCH_VERBOSE ? "inherit" : "ignore",
  });

const waitForPort = (port, timeoutMs = 30_000) =>
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
        setTimeout(tryConnect, 200);
      });
    };
    tryConnect();
  });

const isPortAvailable = (port) =>
  new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => {
      server.close(() => resolve(true));
    });
    server.listen(port, "127.0.0.1");
  });

const findAvailablePort = async (preferredPort) => {
  for (let port = preferredPort; port < preferredPort + 100; port += 1) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }
  throw new Error(`No available port found near ${preferredPort}`);
};

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
    }, 2_000);
  });

const quantile = (values, q) => {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * q) - 1));
  return sorted[index];
};

const summarize = (samples) => ({
  samples_ms: samples.map((value) => Math.round(value)),
  first_ms: Math.round(samples[0] ?? 0),
  warm_median_ms:
    samples.length > 1 ? Math.round(quantile(samples.slice(1), 0.5)) : null,
  min_ms: Math.round(Math.min(...samples)),
  median_ms: Math.round(quantile(samples, 0.5)),
  p95_ms: Math.round(quantile(samples, 0.95)),
  max_ms: Math.round(Math.max(...samples)),
});

const formatResult = (result) => {
  const formatted = {
    ...result,
    button_selector: result.buttonSelector,
    ready_selector: result.readySelector,
  };
  delete formatted.buttonSelector;
  delete formatted.readySelector;
  const budget = benchmarkBudgets[benchMode]?.[result.name];
  if (budget) {
    formatted.budget = {
      warm_median_ms: budget.warmMedianMs,
      p95_ms: budget.p95Ms,
    };
  }
  return formatted;
};

const collectBudgetFailures = (results) => {
  const failures = [];
  for (const result of results) {
    if (result.console_issues.length > 0) {
      failures.push(`${result.name} logged ${result.console_issues.length} console issue(s)`);
    }
    if (result.network_issues.length > 0) {
      failures.push(`${result.name} had ${result.network_issues.length} failed network response(s)`);
    }
    const budget = benchmarkBudgets[benchMode]?.[result.name];
    if (!budget) {
      continue;
    }
    if (result.warm_median_ms !== null && result.warm_median_ms > budget.warmMedianMs) {
      failures.push(
        `${result.name} warm median ${result.warm_median_ms}ms exceeded ${budget.warmMedianMs}ms`
      );
    }
    if (result.p95_ms > budget.p95Ms) {
      failures.push(`${result.name} p95 ${result.p95_ms}ms exceeded ${budget.p95Ms}ms`);
    }
  }
  return failures;
};

async function waitForAppReady(page) {
  await page.waitForSelector(".app-shell", { timeout: 15_000 });
  await page.waitForSelector(".pk-prompt-input-textarea", { timeout: 15_000 });
}

async function benchmarkCase(browser, baseUrl, testCase) {
  const samples = [];
  const consoleIssues = [];
  const networkIssues = [];

  for (let index = 0; index < sampleCount; index += 1) {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    page.on("console", (message) => {
      if (message.type() === "error" || message.type() === "warning") {
        consoleIssues.push({
          sample: index + 1,
          type: message.type(),
          text: message.text().slice(0, 240),
        });
      }
    });
    page.on("response", (response) => {
      if (response.status() >= 400) {
        networkIssues.push({
          sample: index + 1,
          status: response.status(),
          url: response.url().replace(baseUrl, ""),
        });
      }
    });

    const loadStarted = performance.now();
    await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
    await waitForAppReady(page);
    if (testCase.name === "app-ready") {
      samples.push(performance.now() - loadStarted);
      await context.close();
      continue;
    }

    if (settleMs > 0) {
      await page.waitForTimeout(settleMs);
    }

    const button = page.locator(testCase.buttonSelector);
    const buttonCount = await button.count();
    if (buttonCount !== 1) {
      throw new Error(
        `Expected one ${testCase.name} button for ${testCase.buttonSelector}, found ${buttonCount}`
      );
    }
    const started = performance.now();
    await button.click();
    await page.waitForSelector(testCase.readySelector, { timeout: 15_000 });
    samples.push(performance.now() - started);
    await context.close();
  }

  return {
    ...testCase,
    ...summarize(samples),
    console_issues: consoleIssues,
    network_issues: networkIssues,
  };
}

async function main() {
  const mockPort = process.env.BENCH_API_PORT
    ? preferredMockPort
    : await findAvailablePort(preferredMockPort);
  const vitePort = process.env.BENCH_WEB_PORT
    ? preferredVitePort
    : await findAvailablePort(preferredVitePort);
  const baseUrl = `http://127.0.0.1:${vitePort}`;

  const mockApi = startProcess("node", ["scripts/mock-api.mjs"], {
    MOCK_API_PORT: String(mockPort),
  });
  const viteArgs =
    benchMode === "preview"
      ? ["exec", "vite", "preview", "--host", "127.0.0.1", "--port", String(vitePort)]
      : ["exec", "vite", "--host", "127.0.0.1", "--port", String(vitePort)];
  const vite = startProcess("pnpm", viteArgs, {
    VITE_PROXY_API_TARGET: `http://127.0.0.1:${mockPort}`,
  });

  try {
    await waitForPort(mockPort);
    await waitForPort(vitePort);

    const browser = await chromium.launch({ headless: true });
    try {
      const appReady = await benchmarkCase(browser, baseUrl, {
        name: "app-ready",
        buttonSelector: "",
        readySelector: ".app-shell",
      });
      const routeResults = [];
      for (const testCase of cases) {
        routeResults.push(await benchmarkCase(browser, baseUrl, testCase));
      }
      const results = [appReady, ...routeResults];
      const budgetFailures = enforceBudgets ? collectBudgetFailures(results) : [];
      console.log(
        JSON.stringify(
          {
            mode: benchMode,
            enforced: enforceBudgets,
            base_url: baseUrl,
            samples: sampleCount,
            settle_ms: settleMs,
            budget_failures: budgetFailures,
            results: results.map(formatResult),
          },
          null,
          2
        )
      );
      if (budgetFailures.length > 0) {
        throw new Error(`UI benchmark budgets failed:\n${budgetFailures.join("\n")}`);
      }
    } finally {
      await browser.close();
    }
  } finally {
    await Promise.all([stopProcess(vite), stopProcess(mockApi)]);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
