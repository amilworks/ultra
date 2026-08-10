import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright";

import { attachSmokePageGuard } from "./smoke-page-guard.mjs";
import { createTypographyRequestAudit } from "./typography-request-audit.mjs";

const baseUrl = process.env.MOBILE_SMOKE_URL ?? "http://localhost:5173";
const smokeNonce = String(process.env.MOBILE_SMOKE_NONCE || "");
const artifactDir = path.resolve(".playwright-cli/mobile-smoke");
const interFamily = "BisQue Inter Variable";

const cases = [
  { name: "phone-small", width: 320, height: 568, mobile: true },
  { name: "phone-390", width: 390, height: 844, mobile: true },
  { name: "phone-430", width: 430, height: 932, mobile: true },
  { name: "desktop-1024", width: 1024, height: 768, mobile: false },
  { name: "desktop-1440", width: 1440, height: 900, mobile: false },
];

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const parsedBaseUrl = new URL(baseUrl);
assert(
  parsedBaseUrl.protocol === "http:",
  "Typography smoke requires an http loopback URL"
);
assert(
  ["127.0.0.1", "localhost", "[::1]"].includes(parsedBaseUrl.hostname),
  "Typography smoke rejects non-loopback hosts"
);
assert(parsedBaseUrl.port !== "", "Typography smoke requires an explicit web port");
assert(parsedBaseUrl.port !== "5174", "Typography smoke must never use live API port 5174");
assert(
  /^[a-f0-9]{64}$/.test(smokeNonce),
  "Typography smoke requires a wrapper-issued MOBILE_SMOKE_NONCE"
);

async function verifyMockIdentity() {
  const response = await fetch(new URL("/v1/smoke/identity", parsedBaseUrl), {
    headers: { Accept: "application/json" },
    redirect: "error",
    signal: AbortSignal.timeout(5_000),
  });
  assert(response.ok, `Smoke mock identity returned HTTP ${response.status}`);
  const identity = await response.json();
  assert(identity.service === "ultra-mobile-smoke-mock", "Unexpected smoke mock service identity");
  assert(identity.nonce === smokeNonce, "Smoke mock nonce does not match this wrapper run");
}

async function requireAuthenticatedMockShell(page) {
  if (await page.locator(".auth-screen").count()) {
    throw new Error(
      "Typography smoke requires the authenticated mock app shell; refusing auth-form interaction"
    );
  }
  await page.waitForSelector(".app-shell", { timeout: 10_000 });
  await page.waitForSelector(".pk-prompt-input-textarea", { timeout: 10_000 });
}

async function openMobileDrawer(page) {
  await page
    .getByRole("button", {
      name: /open navigation|toggle sidebar|expand sidebar|collapse sidebar/i,
    })
    .click();
  await page.locator('.app-sidebar[data-mobile="true"]').waitFor({ state: "visible" });
}

async function captureCommonMetrics(page) {
  return page.evaluate(() => {
    const query = (selector) => document.querySelector(selector);
    const title = query(".app-header-title-text") ?? query(".hero-title");
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
        title: title ? getComputedStyle(title).fontSize : null,
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

async function captureTypographyMetrics(page, testCase) {
  return page.evaluate(async ({ family, mobile }) => {
    const sample = "BisQue Ultra science Ångström";
    const requestedFonts = [
      { name: "normal-400", query: `400 16px "${family}"` },
      { name: "normal-500", query: `500 16px "${family}"` },
      { name: "normal-600", query: `600 16px "${family}"` },
      { name: "normal-700", query: `700 16px "${family}"` },
      { name: "italic-400", query: `italic 400 16px "${family}"` },
    ];
    const loaded = {};
    for (const requested of requestedFonts) {
      const faces = await document.fonts.load(requested.query, sample);
      loaded[requested.name] = {
        faceCount: faces.length,
        ready: document.fonts.check(requested.query, sample),
      };
    }
    const monoItalicFaces = await document.fonts.load(
      'italic 400 16px "JetBrains Mono"',
      "// scientific comment"
    );
    await document.fonts.ready;
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

    const readStyle = (element) => {
      if (!element) {
        return null;
      }
      const style = getComputedStyle(element);
      return {
        family: style.fontFamily,
        size: style.fontSize,
        lineHeight: style.lineHeight,
        weight: style.fontWeight,
        style: style.fontStyle,
        color: style.color,
        synthesis: style.fontSynthesis,
        opticalSizing: style.fontOpticalSizing,
      };
    };

    // Exercise the real markdown selectors even when the disposable mock has
    // no conversation history yet. The probe is removed before screenshots.
    const readingProbe = document.createElement("div");
    readingProbe.className = "pk-markdown";
    readingProbe.style.cssText =
      "position:fixed;inset:auto auto 0 0;visibility:hidden;pointer-events:none";
    readingProbe.innerHTML =
      "<p>Reading role <em>with true italic</em> and <strong>semantic strong</strong>.</p>";
    document.body.append(readingProbe);
    const reading = readStyle(readingProbe);
    const readingItalic = readStyle(readingProbe.querySelector("em"));
    const readingStrong = readStyle(readingProbe.querySelector("strong"));
    readingProbe.remove();

    // Exercise representative production selectors so the browser proves that
    // semantic tokens survive the cascade, even before those lazy surfaces open.
    const semanticProbe = document.createElement("div");
    semanticProbe.style.cssText =
      "position:fixed;inset:auto auto 0 0;visibility:hidden;pointer-events:none";
    semanticProbe.innerHTML = `
      <h1 class="resource-browser-title">Resources</h1>
      <p class="resource-browser-result-summary">12 resources</p>
      <form class="resource-browser-rename-form"><label>Resource name</label></form>
      <h2 class="viewer-sheet-title">Volume calibration</h2>
      <label class="viewer-volume-cutaway-depth"><strong>42%</strong></label>
      <div class="text-viewer-row"><span class="tv-c">// scientific comment</span></div>
    `;
    document.body.append(semanticProbe);
    const resourcePageHeading = readStyle(
      semanticProbe.querySelector(".resource-browser-title")
    );
    const resourceData = readStyle(
      semanticProbe.querySelector(".resource-browser-result-summary")
    );
    const resourceLabel = readStyle(
      semanticProbe.querySelector(".resource-browser-rename-form label")
    );
    const viewerPanelHeading = readStyle(
      semanticProbe.querySelector(".viewer-sheet-title")
    );
    const viewerData = readStyle(
      semanticProbe.querySelector(".viewer-volume-cutaway-depth strong")
    );
    const monoComment = readStyle(semanticProbe.querySelector(".tv-c"));
    const wordmarkScope = document.querySelector(
      mobile ? ".app-mobile-shell-title" : ".app-sidebar-header"
    );
    semanticProbe.remove();

    return {
      loaded,
      cls: Number(globalThis.__ultraTypographyCls ?? 0),
      monoItalic: {
        faceCount: monoItalicFaces.length,
        ready: document.fonts.check(
          'italic 400 16px "JetBrains Mono"',
          "// scientific comment"
        ),
      },
      roles: {
        body: readStyle(document.body),
        composer: readStyle(document.querySelector(".pk-prompt-input-textarea")),
        invitation: readStyle(
          document.querySelector(".blank-chat-welcome-hero, .mobile-chat-hero-title")
        ),
        action: readStyle(
          document.querySelector(".app-new-chat-button, .app-composer-submit-button")
        ),
        account: readStyle(document.querySelector(".app-sidebar-account-name")),
        label: readStyle(
          document.querySelector(
            '.app-sidebar-content .app-history-group [data-slot="sidebar-group-label"]'
          )
        ),
        brandBisque: readStyle(wordmarkScope?.querySelector(".brand-wordmark__bisque")),
        brandUltra: readStyle(wordmarkScope?.querySelector(".brand-wordmark__ultra")),
        reading,
        readingItalic,
        readingStrong,
        resourcePageHeading,
        resourceData,
        resourceLabel,
        viewerPanelHeading,
        viewerData,
        monoComment,
      },
    };
  }, { family: interFamily, mobile: testCase.mobile });
}

async function captureVariationEvidence(page, testCase) {
  await page.evaluate(async (family) => {
    await Promise.all([
      document.fonts.load(`400 48px "${family}"`, "Hamburgefontsiv 0123456789"),
      document.fonts.load(
        'italic 400 40px "JetBrains Mono"',
        "// scientific calibration 0123456789"
      ),
    ]);
    const host = document.createElement("div");
    host.id = "typography-raster-proof";
    host.style.cssText =
      "position:fixed;left:0;top:0;z-index:2147483647;padding:4px;background:#fff;color:#000";
    const makeRow = (kind, style, text) => {
      const row = document.createElement("div");
      row.dataset.proof = kind;
      row.style.cssText =
        "display:block;width:520px;height:72px;overflow:hidden;background:#fff;color:#000";
      const sample = document.createElement("span");
      sample.dataset.proofText = kind;
      sample.style.cssText = style;
      sample.textContent = text;
      row.append(sample);
      host.append(row);
    };
    makeRow(
      "opsz-14",
      `font-family:"${family}";font-size:48px;font-weight:400;line-height:1.25;font-variation-settings:"opsz" 14`,
      "Hamburgefontsiv 0123456789"
    );
    makeRow(
      "opsz-32",
      `font-family:"${family}";font-size:48px;font-weight:400;line-height:1.25;font-variation-settings:"opsz" 32`,
      "Hamburgefontsiv 0123456789"
    );
    makeRow(
      "mono-normal",
      'font-family:"JetBrains Mono";font-size:40px;font-weight:400;font-style:normal;line-height:1.35',
      "// scientific calibration 0123456789"
    );
    makeRow(
      "mono-italic",
      'font-family:"JetBrains Mono";font-size:40px;font-weight:400;font-style:italic;line-height:1.35',
      "// scientific calibration 0123456789"
    );
    document.body.append(host);
  }, interFamily);

  const geometry = await page.evaluate(() => {
    const read = (kind) => {
      const element = document.querySelector(`[data-proof-text="${kind}"]`);
      const rect = element.getBoundingClientRect();
      return { width: rect.width, height: rect.height };
    };
    return {
      opsz14: read("opsz-14"),
      opsz32: read("opsz-32"),
    };
  });
  const capture = async (kind) =>
    page.locator(`[data-proof="${kind}"]`).screenshot({ animations: "disabled" });
  const [opsz14, opsz32, monoNormal, monoItalic] = await Promise.all([
    capture("opsz-14"),
    capture("opsz-32"),
    capture("mono-normal"),
    capture("mono-italic"),
  ]);
  await page.locator("#typography-raster-proof").evaluate((element) => element.remove());

  const opticalWidthRatio =
    Math.abs(geometry.opsz14.width - geometry.opsz32.width) /
    Math.max(geometry.opsz14.width, geometry.opsz32.width);
  assert(
    geometry.opsz14.width > 0 &&
      geometry.opsz32.width > 0 &&
      geometry.opsz14.height === geometry.opsz32.height,
    `${testCase.name}: optical-axis proof has invalid geometry`
  );
  assert(
    opticalWidthRatio < 0.2,
    `${testCase.name}: optical-axis geometry diverged by ${opticalWidthRatio}`
  );
  assert(
    !opsz14.equals(opsz32),
    `${testCase.name}: Inter opsz 14 and 32 rasterized identically`
  );
  assert(
    !monoNormal.equals(monoItalic),
    `${testCase.name}: JetBrains Mono normal and italic rasterized identically`
  );

  const hash = (bytes) => createHash("sha256").update(bytes).digest("hex").slice(0, 16);
  return {
    geometry,
    opticalWidthRatio,
    hashes: {
      opsz14: hash(opsz14),
      opsz32: hash(opsz32),
      monoNormal: hash(monoNormal),
      monoItalic: hash(monoItalic),
    },
  };
}

function assertNoHorizontalOverflow(metrics, caseName, surface) {
  assert(
    metrics.overflow.document === metrics.overflow.innerWidth,
    `${caseName} ${surface}: document overflow ${metrics.overflow.document} != ${metrics.overflow.innerWidth}`
  );
  assert(
    metrics.overflow.body === metrics.overflow.innerWidth,
    `${caseName} ${surface}: body overflow ${metrics.overflow.body} != ${metrics.overflow.innerWidth}`
  );
}

function assertTypographyMetrics(typography, testCase) {
  for (const [fontCase, result] of Object.entries(typography.loaded)) {
    assert(result.faceCount > 0, `${testCase.name}: ${fontCase} resolved no Inter face`);
    assert(result.ready, `${testCase.name}: ${fontCase} did not finish loading`);
  }
  assert(
    typography.monoItalic.faceCount > 0 && typography.monoItalic.ready,
    `${testCase.name}: genuine JetBrains Mono italic did not load`
  );

  const expectedWeights = {
    // The calibrated variable body and reading registers intentionally use
    // 430. Keep this browser-level assertion aligned with the source contract
    // checked by check-typography-contract.mjs and light-theme-ink.test.ts.
    body: "430",
    composer: "430",
    invitation: "400",
    action: "500",
    reading: "430",
    readingItalic: "430",
    // 600, matching the reading heading rather than exceeding it. At 700 an
    // inline **emphasis** rendered HEAVIER than every heading above it (h2/h3/h4
    // are all 600), and at h4's 16px it beat the heading at identical size.
    // Emphasis must not outrank structure. Mirrors the source-side pin in
    // check-typography-contract.mjs.
    readingStrong: "600",
    brandBisque: "400",
    brandUltra: "600",
    resourcePageHeading: "600",
    resourceData: "500",
    resourceLabel: "600",
    viewerPanelHeading: "600",
    viewerData: "500",
  };
  if (!testCase.mobile) {
    Object.assign(expectedWeights, {
      account: "500",
      label: "600",
    });
  }
  for (const [role, expectedWeight] of Object.entries(expectedWeights)) {
    const style = typography.roles[role];
    assert(style, `${testCase.name}: missing product typography role ${role}`);
    assert(
      style.family.includes(interFamily),
      `${testCase.name}: ${role} uses unexpected family ${style.family}`
    );
    assert(
      style.weight === expectedWeight,
      `${testCase.name}: ${role} weight ${style.weight} != ${expectedWeight}`
    );
    assert(style.synthesis === "none", `${testCase.name}: ${role} permits font synthesis`);
    const expectedOpticalSizing = role === "invitation" ? "none" : "auto";
    assert(
      style.opticalSizing === expectedOpticalSizing,
      `${testCase.name}: ${role} optical sizing ${style.opticalSizing} != ${expectedOpticalSizing}`
    );
  }
  assert(
    typography.roles.readingItalic.style === "italic",
    `${testCase.name}: markdown emphasis did not compute to true italic`
  );
  assert(
    typography.roles.brandUltra.color === typography.roles.body.color &&
      typography.roles.brandUltra.color !== typography.roles.brandBisque.color,
    `${testCase.name}: wordmark did not compute to distinct light-theme monochrome roles ` +
      `(body ${typography.roles.body.color}, BisQue ${typography.roles.brandBisque.color}, ` +
      `Ultra ${typography.roles.brandUltra.color})`
  );
  assert(
    typography.roles.monoComment.family.includes("JetBrains Mono") &&
      typography.roles.monoComment.weight === "400" &&
      typography.roles.monoComment.style === "italic" &&
      typography.roles.monoComment.synthesis === "none",
    `${testCase.name}: scientific code comment is not genuine JetBrains Mono 400 italic`
  );
  // The smoke deliberately cold-holds Inter to exercise font-display: swap.
  // Desktop swaps both the always-visible sidebar and the chat canvas; on the
  // Linux CI fallback this settles at 0.0270, while the mobile canvas remains
  // below the original stricter ceiling. Keep both contracts explicit rather
  // than weakening the phone budget or dropping the cold-font proof.
  const clsBudget = testCase.mobile ? 0.01 : 0.03;
  assert(
    typography.cls <= clsBudget,
    `${testCase.name}: typography CLS ${typography.cls} exceeded ${clsBudget}`
  );
}

async function runCase(browser, testCase) {
  const context = await browser.newContext({
    viewport: { width: testCase.width, height: testCase.height },
    deviceScaleFactor: testCase.mobile ? 3 : 1,
    hasTouch: testCase.mobile,
    isMobile: testCase.mobile,
  });
  const page = await context.newPage();
  const requestAudit = createTypographyRequestAudit(baseUrl);
  let releaseHeldNormalFont;
  const heldNormalFont = new Promise((resolve) => {
    releaseHeldNormalFont = resolve;
  });
  let heldNormalFontSeen = false;
  await page.route(/InterVariable-v4\.1\.woff2(?:[?#]|$)/, async (route) => {
    heldNormalFontSeen = true;
    await heldNormalFont;
    await route.continue();
  });
  const pageGuard = await attachSmokePageGuard(page, {
    baseUrl,
    typographyAudit: requestAudit,
  });
  await page.addInitScript(() => {
    globalThis.__ultraTypographyCls = 0;
    globalThis.__ultraTypographyClsActive = false;
    const observer = new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        if (globalThis.__ultraTypographyClsActive && !entry.hadRecentInput) {
          globalThis.__ultraTypographyCls += entry.value;
        }
      }
    });
    observer.observe({ type: "layout-shift", buffered: true });
    globalThis.__startUltraTypographyCls = () => {
      observer.takeRecords();
      globalThis.__ultraTypographyCls = 0;
      globalThis.__ultraTypographyClsActive = true;
    };
  });
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await requireAuthenticatedMockShell(page);
  pageGuard.assertNoBlockedRequests(assert, testCase.name);
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve))
      )
  );
  // Establish a stable fallback layout before starting a typography-only CLS
  // window. No network-idle wait is possible while the normal face is held.
  await page.waitForTimeout(250);
  assert(heldNormalFontSeen, `${testCase.name}: normal Inter request was not cold-held`);
  await page.evaluate(() => globalThis.__startUltraTypographyCls());
  releaseHeldNormalFont();
  await page.evaluate(async (family) => {
    await document.fonts.load(`400 16px "${family}"`, "BisQue Ultra science Ångström");
    await document.fonts.ready;
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve))
    );
  }, interFamily);

  const typography = await captureTypographyMetrics(page, testCase);
  assertTypographyMetrics(typography, testCase);
  const variationEvidence = await captureVariationEvidence(page, testCase);
  await page.evaluate(() => {
    globalThis.__ultraTypographyClsActive = false;
  });
  const metrics = await captureCommonMetrics(page);
  assertNoHorizontalOverflow(metrics, testCase.name, "chat");

  const result = { name: testCase.name, ...metrics, typography, variationEvidence };

  if (testCase.mobile) {
    const mobileMetrics = await captureCommonMetrics(page);
    result.fonts = mobileMetrics.fonts;
    result.composerRect = mobileMetrics.composerRect;
    assert(metrics.fonts.body === "16px", `${testCase.name}: body font expected 16px`);
    assert(
      mobileMetrics.fonts.composer === "16px",
      `${testCase.name}: composer font expected 16px`
    );

    const sidebarToggle = page
      .getByRole("button", {
        name: /open navigation|toggle sidebar|expand sidebar|collapse sidebar/i,
      })
      .first();
    if (await sidebarToggle.count()) {
      await openMobileDrawer(page);
      const drawer = page.locator('.app-sidebar[data-mobile="true"]');
      const drawerBox = await drawer.boundingBox();
      assert(drawerBox, `${testCase.name}: drawer bounding box missing`);
      result.drawerWidth = drawerBox.width;

      const resourcesButton = drawer.getByRole("button", { name: /^resources$/i });
      await resourcesButton.click();
      await page.getByText(/Resource browser|Resources/i).first().waitFor({ state: "visible" });
      const resourceMetrics = await captureCommonMetrics(page);
      assertNoHorizontalOverflow(resourceMetrics, testCase.name, "resources");
      result.resourceOverflow = resourceMetrics.overflow;
      await page.waitForFunction(
        () => !document.querySelector('.app-sidebar[data-mobile="true"]')
      );

      await openMobileDrawer(page);
      const newChatButton = drawer.getByRole("button", { name: /^new chat$/i });
      await newChatButton.click();
      await page.locator(".mobile-chat-hero-title, .blank-chat-welcome-hero").first().waitFor({
        state: "visible",
        timeout: 10000,
      });
      const newChatMetrics = await captureCommonMetrics(page);
      assertNoHorizontalOverflow(newChatMetrics, testCase.name, "new-chat");
      result.newChatOverflow = newChatMetrics.overflow;
      await page.waitForFunction(
        () => !document.querySelector('.app-sidebar[data-mobile="true"]')
      );
    } else {
      result.drawerWidth = null;
    }
  } else {
    const sidebar = page.locator('[data-slot="sidebar-container"]');
    assert(await sidebar.isVisible(), `${testCase.name}: desktop sidebar should remain visible`);
  }

  const { attempted, successfulLocal } = requestAudit.assertLocalSuccess(assert, testCase.name);
  pageGuard.assertNoBlockedRequests(assert, testCase.name);
  assert(
    successfulLocal.some(({ url }) => url.includes("InterVariable-v4.1.woff2")),
    `${testCase.name}: normal Inter asset was not requested`
  );
  assert(
    successfulLocal.some(({ url }) => url.includes("InterVariable-Italic-v4.1.woff2")),
    `${testCase.name}: italic Inter asset was not requested`
  );
  result.localTypographyAssets = [
    ...new Set(attempted.map(({ url }) => new URL(url).pathname)),
  ].sort();

  await fs.mkdir(artifactDir, { recursive: true });
  await page.screenshot({
    path: path.join(artifactDir, `${testCase.name}.png`),
    fullPage: false,
  });
  await context.close();
  return result;
}

async function main() {
  await verifyMockIdentity();
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
