import fs from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright";

const baseUrl = process.env.MOBILE_SMOKE_URL ?? "http://localhost:5173";
const artifactDir = path.resolve(".playwright-cli/mobile-smoke");

const cases = [
  { name: "phone-small", width: 320, height: 568, mobile: true },
  { name: "phone-390", width: 390, height: 844, mobile: true },
  { name: "phone-430", width: 430, height: 932, mobile: true },
  { name: "desktop-1440", width: 1440, height: 900, mobile: false },
];

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

async function maybeContinueAsGuest(page) {
  const authScreen = page.locator(".auth-screen");
  if (!(await authScreen.count())) {
    return;
  }

  const guestNameInput = page.locator("#guest-name");
  if (!(await guestNameInput.count())) {
    const guestTab = page.getByRole("button", { name: /^continue as guest$/i }).first();
    if (await guestTab.count()) {
      await guestTab.click();
    }
  }

  if (await guestNameInput.count()) {
    await guestNameInput.fill("Mobile Smoke");
    await page.locator("#guest-email").fill("mobile.smoke@example.com");
    await page.locator("#guest-affiliation").fill("BisQue Ultra QA");
    await page.locator("form.auth-form button[type='submit']").click();
    await page.waitForLoadState("networkidle");
  }
}

async function openMobileDrawer(page) {
  await page.getByRole("button", { name: /toggle sidebar/i }).click();
  await page.locator('.app-sidebar[data-mobile="true"]').waitFor({ state: "visible" });
}

async function captureCommonMetrics(page) {
  return page.evaluate(() => {
    const query = (selector) => document.querySelector(selector);
    const headerTitle = query(".app-header-title-text");
    const composer = query(".app-composer-textarea");
    const bodyStyles = getComputedStyle(document.body);
    const composerStyles = composer ? getComputedStyle(composer) : null;
    return {
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
        dpr: window.devicePixelRatio,
      },
      overflow: {
        document: document.documentElement.scrollWidth,
        body: document.body.scrollWidth,
        innerWidth: window.innerWidth,
      },
      fonts: {
        body: bodyStyles.fontSize,
        headerTitle: headerTitle ? getComputedStyle(headerTitle).fontSize : null,
        composer: composerStyles?.fontSize ?? null,
      },
      composerRect: composer
        ? (() => {
            const rect = composer.getBoundingClientRect();
            return {
              top: rect.top,
              bottom: rect.bottom,
              left: rect.left,
              right: rect.right,
              width: rect.width,
              height: rect.height,
            };
          })()
        : null,
    };
  });
}

async function runCase(browser, testCase) {
  const context = await browser.newContext({
    viewport: { width: testCase.width, height: testCase.height },
    deviceScaleFactor: testCase.mobile ? 3 : 1,
    hasTouch: testCase.mobile,
    isMobile: testCase.mobile,
  });
  const page = await context.newPage();
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await maybeContinueAsGuest(page);
  await page.waitForLoadState("networkidle");
  await page.waitForSelector(".app-header-title-text", { timeout: 10000 });

  const metrics = await captureCommonMetrics(page);
  assert(
    metrics.overflow.document === metrics.overflow.innerWidth,
    `${testCase.name}: document overflow ${metrics.overflow.document} != ${metrics.overflow.innerWidth}`
  );
  assert(
    metrics.overflow.body === metrics.overflow.innerWidth,
    `${testCase.name}: body overflow ${metrics.overflow.body} != ${metrics.overflow.innerWidth}`
  );

  const result = { name: testCase.name, ...metrics };

  if (testCase.mobile) {
    await page.waitForSelector(".pk-prompt-input-textarea", { timeout: 10000 });
    const mobileMetrics = await captureCommonMetrics(page);
    result.fonts = mobileMetrics.fonts;
    result.composerRect = mobileMetrics.composerRect;
    assert(metrics.fonts.body === "16px", `${testCase.name}: body font expected 16px`);
    assert(
      mobileMetrics.fonts.composer === "16px",
      `${testCase.name}: composer font expected 16px`
    );

    await openMobileDrawer(page);
    const drawer = page.locator('.app-sidebar[data-mobile="true"]');
    const drawerBox = await drawer.boundingBox();
    assert(drawerBox, `${testCase.name}: drawer bounding box missing`);
    result.drawerWidth = drawerBox.width;

    const resourcesButton = drawer.getByRole("button", { name: /^resources$/i });
    await resourcesButton.click();
    await page.waitForLoadState("networkidle");
    await page
      .locator(".app-header-title-text")
      .filter({ hasText: "Resource browser" })
      .waitFor({ state: "visible" });
    await page.waitForFunction(
      () => !document.querySelector('.app-sidebar[data-mobile="true"]')
    );

    await openMobileDrawer(page);
    const newChatButton = drawer.getByRole("button", { name: /new chat/i });
    await newChatButton.click();
    await page.waitForLoadState("networkidle");
    await page
      .locator(".app-header-title-text")
      .filter({ hasText: "New conversation" })
      .waitFor({ state: "visible" });
    await page.waitForFunction(
      () => !document.querySelector('.app-sidebar[data-mobile="true"]')
    );
  } else {
    const sidebar = page.locator('[data-slot="sidebar-container"]');
    assert(await sidebar.isVisible(), "desktop-1440: desktop sidebar should remain visible");
  }

  await fs.mkdir(artifactDir, { recursive: true });
  await page.screenshot({
    path: path.join(artifactDir, `${testCase.name}.png`),
    fullPage: false,
  });
  await context.close();
  return result;
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  try {
    const results = [];
    for (const testCase of cases) {
      results.push(await runCase(browser, testCase));
    }
    console.log(JSON.stringify({ baseUrl, results }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
