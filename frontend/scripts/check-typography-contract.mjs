import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const frontendRoot = path.resolve(import.meta.dirname, "..");
const sourceRoot = path.join(frontendRoot, "src");
const distRoot = path.join(frontendRoot, "dist");
const checkDist = process.argv.includes("--dist");
const failures = [];

const expectedFonts = [
  {
    file: "InterVariable-v4.1.woff2",
    style: "normal",
    bytes: 352_240,
    sha256: "693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3",
    url: "https://rsms.me/inter/font-files/InterVariable.woff2?v=4.1",
  },
  {
    file: "InterVariable-Italic-v4.1.woff2",
    style: "italic",
    bytes: 387_976,
    sha256: "e564f652916db6c139570fefb9524a77c4d48f30c92928de9db19b6b5c7a262a",
    url: "https://rsms.me/inter/font-files/InterVariable-Italic.woff2?v=4.1",
  },
];

const read = (relativePath) => fs.readFileSync(path.join(frontendRoot, relativePath), "utf8");
const digest = (buffer) => crypto.createHash("sha256").update(buffer).digest("hex");
const check = (condition, message) => {
  if (!condition) {
    failures.push(message);
  }
};

const walkFiles = (root) => {
  if (!fs.existsSync(root)) {
    return [];
  }
  return fs.readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(root, entry.name);
    return entry.isDirectory() ? walkFiles(entryPath) : [entryPath];
  });
};

const cssBlocks = (css, atRule) => {
  const pattern = new RegExp(`${atRule}\\s*\\{[^}]*\\}`, "gi");
  return css.match(pattern) ?? [];
};

const contrastRatio = (foreground, background) => {
  const luminance = (hex) => {
    const channels = hex
      .replace("#", "")
      .match(/.{2}/g)
      .map((channel) => Number.parseInt(channel, 16) / 255)
      .map((channel) =>
        channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4
      );
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
  };
  const first = luminance(foreground);
  const second = luminance(background);
  return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05);
};

const typographyCss = read("src/typography.css");
const stylesCss = read("src/styles.css");
const mainSource = read("src/main.tsx");
const appSource = read("src/App.tsx");
const authSource = read("src/components/auth/AuthScreen.tsx");
const viewerPageSource = read("src/components/ScientificViewerPage.tsx");
const wordmarkSource = read("src/components/BrandWordmark.tsx");
const carpetSource = read("src/components/viewer/cifti/CiftiCarpet.tsx");
const connectivitySource = read("src/components/viewer/cifti/CiftiConnectivity.tsx");
const packageSource = read("package.json");
const lockSource = read("pnpm-lock.yaml");
const smokeRunnerSource = read("scripts/run-mobile-smoke.mjs");
const mobileSmokeSource = read("scripts/mobile-smoke.mjs");
const mockApiSource = read("scripts/mock-api.mjs");
const provenance = read("src/assets/fonts/PROVENANCE.md");
const dockerfile = read("Dockerfile");
const caddyPath = path.resolve(
  frontendRoot,
  "..",
  "deploy/caddy/Caddyfile.single-host.template"
);
const caddyfile = fs.existsSync(caddyPath) ? fs.readFileSync(caddyPath, "utf8") : null;

const sourceFontDir = path.join(sourceRoot, "assets", "fonts");
const sourceFontFiles = walkFiles(sourceFontDir);
const sourceFontBinaries = sourceFontFiles.filter((file) => /\.(?:woff2?|ttf|otf)$/i.test(file));
check(sourceFontBinaries.length === 2, `Expected two source font binaries, found ${sourceFontBinaries.length}`);
check(
  sourceFontBinaries.every((file) => file.endsWith(".woff2")),
  "Source font assets must use WOFF2 only"
);

for (const font of expectedFonts) {
  const fontPath = path.join(sourceFontDir, font.file);
  check(fs.existsSync(fontPath), `Missing source font: ${font.file}`);
  if (fs.existsSync(fontPath)) {
    const bytes = fs.readFileSync(fontPath);
    check(bytes.length === font.bytes, `${font.file} has ${bytes.length} bytes; expected ${font.bytes}`);
    check(digest(bytes) === font.sha256, `${font.file} SHA-256 does not match official v4.1`);
  }
  for (const provenanceValue of [
    font.file,
    font.style,
    font.url,
    String(font.bytes),
    font.sha256,
    "`wght` 100–900",
    "`opsz` 14–32",
  ]) {
    check(provenance.includes(provenanceValue), `Provenance is missing ${provenanceValue}`);
  }
}

const licensePath = path.join(sourceFontDir, "OFL-1.1.txt");
check(fs.existsSync(licensePath), "Missing Inter OFL-1.1 license");
if (fs.existsSync(licensePath)) {
  const licenseBytes = fs.readFileSync(licensePath);
  check(licenseBytes.length === 4_380, "Inter OFL-1.1 license byte count changed");
  check(
    digest(licenseBytes) === "262481e844521b326f5ecd053e59b98c8b2da78c8ee1bdbb6e8174305e54935a",
    "Inter OFL-1.1 license is not the verbatim v4.1 upstream notice"
  );
}

const sourceFaces = cssBlocks(typographyCss, "@font-face").filter((block) =>
  block.includes("BisQue Inter Variable")
);
check(sourceFaces.length === 2, `Expected two Inter @font-face rules, found ${sourceFaces.length}`);
for (const font of expectedFonts) {
  const face = sourceFaces.find((block) => new RegExp(`font-style:\\s*${font.style}`).test(block));
  check(Boolean(face), `Missing ${font.style} Inter face`);
  if (face) {
    check(face.includes(font.file), `${font.style} face does not reference ${font.file}`);
    check(/font-weight:\s*100 900\s*;/.test(face), `${font.style} face must expose weight 100–900`);
    check(/font-display:\s*swap\s*;/.test(face), `${font.style} face must use font-display: swap`);
    check(/format\(["']woff2["']\)/.test(face), `${font.style} face must declare WOFF2`);
  }
}
check(!/\blocal\s*\(/i.test(typographyCss), "Typography CSS must not bypass vendored faces with local()");
check(!/https?:\/\//i.test(typographyCss), "Typography CSS must not load fonts from the network");
check(/font-optical-sizing:\s*auto\s*;/.test(typographyCss), "Optical sizing must remain automatic");
check(/font-synthesis:\s*none\s*;/.test(typographyCss), "Synthetic bold/italic must remain disabled");

const typographyImport = mainSource.indexOf('import "./typography.css";');
const stylesImport = mainSource.indexOf('import "./styles.css";');
check(typographyImport >= 0 && typographyImport < stylesImport, "Typography CSS must load before styles.css");
check(!mainSource.includes("@fontsource/inter"), "Production entrypoint still imports @fontsource/inter");
check(!packageSource.includes("@fontsource/inter"), "package.json still depends on @fontsource/inter");
check(!lockSource.includes("@fontsource/inter"), "pnpm lockfile still contains @fontsource/inter");
check(
  mainSource.includes('@fontsource/jetbrains-mono/latin-400-italic.css') &&
    mainSource.includes('@fontsource/jetbrains-mono/latin-ext-400-italic.css'),
  "Scientific code comments require genuine JetBrains Mono 400 italic for Latin and Latin-ext"
);
check(
  /\.tv-c\s*\{[^}]*font-style:\s*italic;/s.test(stylesCss),
  "Scientific code-comment syntax must remain italic"
);
check(
  /benefit\s+has not yet been measured/.test(provenance) &&
    /italic face must remain\s+demand-loaded/.test(provenance),
  "Provenance must record the unmeasured no-preload tradeoff and demand-loaded italic policy"
);

const productionCssFiles = walkFiles(sourceRoot).filter(
  (file) => file.endsWith(".css") && path.basename(file) !== "typography.css"
);
for (const cssFile of productionCssFiles) {
  const css = fs.readFileSync(cssFile, "utf8");
  for (const match of css.matchAll(/font-weight\s*:\s*(\d+)\s*;/g)) {
    check(
      ["400", "500", "600", "700"].includes(match[1]),
      `${path.relative(frontendRoot, cssFile)} has noncanonical font-weight ${match[1]}`
    );
  }
}

for (const [token, weight] of [
  // 440, not 400. Inter's stroke-to-letter ratio sets lighter than the
  // grotesques Ultra was measured against: stem/x-height 0.1609 at w400 versus
  // Söhne Buch's 0.1721, driven by Inter's larger x-height rather than a
  // thinner stem. The matching weight solves to 430 by two independent
  // methods (outline geometry and integrated rendered ink); 440 is one step
  // past; 430 is the match itself, adopted after living with 440.
  ["body", "430"],
  ["reading-body", "430"],
  ["nav", "500"],
  ["action", "500"],
  ["data", "500"],
  ["label", "600"],
  ["panel-heading", "600"],
  ["page-heading", "600"],
  ["reading-heading", "600"],
  // 670, dropped from 700 once body moved to 440 narrowed the gap to 260 and
  // the hero title read as a shout. Still outranks reading-heading.
  ["strong", "670"],
  // 600, matching reading-heading rather than exceeding it. At 700 an inline
  // **emphasis** in an answer outweighed every heading above it (h2/h3/h4 are
  // all 600), and at h4's 16px it beat the heading at identical size. Emphasis
  // must not outrank structure.
  ["reading-strong", "600"],
]) {
  check(
    new RegExp(`--font-weight-${token}:\\s*${weight};`).test(stylesCss),
    `Root typography role --font-weight-${token} must be ${weight}`
  );
}

for (const [pattern, message] of [
  [/--font-size-body:\s*0\.9375rem;/, "Desktop body must remain 15px"],
  [/--line-height-body:\s*1\.5;/, "Desktop body line-height must remain 1.5"],
  // 15px, matching --font-size-body. NOTE the 0.9375rem literal is what scopes
  // this to the desktop :root — these patterns run over the whole file, and the
  // phone override below is still 1rem, so a 1rem pattern would match THAT and
  // pass for the wrong reason.
  [/--font-size-reading:\s*0\.9375rem;/, "Desktop chat reading must remain 15px"],
  [/--line-height-reading:\s*1\.62;/, "Desktop chat reading line-height must remain 1.62"],
  [/--user-chat-width:\s*49rem;/, "Desktop chat reading measure must remain 49rem"],
  [/@media \(max-width: 640px\)[\s\S]*--line-height-reading:\s*1\.68;/, "Phone reading line-height must remain 1.68"],
  [/@media \(max-width: 640px\)[\s\S]*--font-size-body:\s*1rem;/, "Phone body must remain 16px"],
  [/@media \(max-width: 640px\)[\s\S]*--font-size-reading:\s*1rem;/, "Phone reading must remain 16px"],
  [/\.pk-prompt-input-textarea\s*\{[^}]*font:\s*inherit;/s, "Composer must inherit the 16px phone font"],
]) {
  check(pattern.test(stylesCss), message);
}

check(appSource.includes("<BrandWordmark"), "Sidebar must use the shared BrandWordmark");
check(authSource.includes("<BrandWordmark"), "Authentication wordmark must use the shared BrandWordmark");
check(
  /app-mobile-shell-title[\s\S]*mobileShellTitle \?\? <BrandWordmark \/>/.test(appSource),
  "Mobile app chrome must use the shared BrandWordmark for the product title"
);
check(
  viewerPageSource.includes("<BrandWordmark") &&
    viewerPageSource.includes('aria-label="Breadcrumb"') &&
    viewerPageSource.includes('aria-current="page"') &&
    viewerPageSource.includes("<h1"),
  "Scientific viewer must expose a branded breadcrumb and h1 page hierarchy"
);
check(
  /\.brand-wordmark__bisque\s*\{[^}]*color:\s*var\(--brand-wordmark-context\);[^}]*font-weight:\s*400;/s.test(
    typographyCss
  ) &&
    /\.brand-wordmark__ultra\s*\{[^}]*color:\s*var\(--brand-wordmark-emphasis\);[^}]*font-weight:\s*600;/s.test(
      typographyCss
    ),
  "Wordmark must keep BisQue context 400 and Ultra emphasis 600"
);
check(
  wordmarkSource.includes('aria-label": ariaLabel = "BisQue Ultra"') &&
    wordmarkSource.includes('role="img"') &&
    (wordmarkSource.match(/aria-hidden="true"/g) ?? []).length === 2,
  "Wordmark must expose one accessible name and hide its split visual spans"
);

// Meridian ladder rungs (emphasis = m0, context = m1), measured against the
// SIDEBAR ground the wordmark sits on in each theme — not white/black, which
// no Meridian surface is. Keep in sync with src/typography.css and the ladder
// pins in src/features/light-theme-ink.test.ts.
for (const [token, expected, background] of [
  ["context-light", "#424547", "#e9ebeb"],
  ["emphasis-light", "#171b1d", "#e9ebeb"],
  ["context-dark", "#a5abb0", "#0f1214"],
  ["emphasis-dark", "#dce3ea", "#0f1214"],
]) {
  const value = typographyCss.match(
    new RegExp(`--brand-wordmark-${token}:\\s*(#[0-9a-f]{6})`, "i")
  )?.[1];
  check(
    value === expected && contrastRatio(value, background) >= 4.5,
    `Wordmark ${token} must be ${expected} and meet AA`
  );
}
check(
  /:root\s*\{[^}]*--brand-wordmark-context:\s*var\(--brand-wordmark-context-light\);[^}]*--brand-wordmark-emphasis:\s*var\(--brand-wordmark-emphasis-light\);/s.test(
    typographyCss
  ),
  "Light wordmark aliases must use monochrome context and emphasis"
);
check(
  /\.dark\s*\{[^}]*--brand-wordmark-context:\s*var\(--brand-wordmark-context-dark\);[^}]*--brand-wordmark-emphasis:\s*var\(--brand-wordmark-emphasis-dark\);/s.test(
    typographyCss
  ),
  "Dark wordmark aliases must use monochrome context and emphasis"
);
check(
  /\.auth-screen-logo\s*\{[^}]*--brand-wordmark-context:\s*var\(--brand-wordmark-context-dark\);[^}]*--brand-wordmark-emphasis:\s*var\(--brand-wordmark-emphasis-dark\);/s.test(
    typographyCss
  ),
  "Authentication wordmark must override both roles for its dark hero"
);

check(/location \^~ \/assets\/ \{/.test(dockerfile), "nginx /assets/ cache location must use ^~");
if (caddyfile === null) {
  console.log("Skipping Caddy typography assertion: deploy/caddy is outside this build context.");
} else {
  check(
    /@staticfiles\s*\{[^}]*not path \/assets\/\*/s.test(caddyfile),
    "Caddy stable-font matcher must exclude hashed /assets/"
  );
}

for (const [selector, token] of [
  [".app-settings-panel-heading h2", "panel-heading"],
  [".resource-browser-title", "page-heading"],
  [".resource-browser-result-summary", "data"],
  [".resource-browser-rename-form label", "label"],
  [".viewer-sheet-title", "panel-heading"],
  [".viewer-volume-cutaway-depth strong", "data"],
  [".viewer-volume-geometry-panel strong", "data"],
  [".viewer-slider-field-head strong", "data"],
  [".training-header-title h2", "page-heading"],
]) {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  check(
    new RegExp(
      `${escapedSelector}\\s*\\{[^}]*font-weight:\\s*var\\(--font-weight-${token}\\)`,
      "s"
    ).test(stylesCss),
    `${selector} must consume --font-weight-${token}`
  );
}

check(
  carpetSource.includes('600 11px "BisQue Inter Variable"') &&
    carpetSource.includes('400 11px "JetBrains Mono"') &&
    carpetSource.includes('400 10px "JetBrains Mono"') &&
    carpetSource.includes("scheduleFontsReadyRedraw"),
  "CIFTI carpet must redraw after all exact Inter and JetBrains canvas fonts resolve"
);
check(
  connectivitySource.includes('600 10px "BisQue Inter Variable"') &&
    connectivitySource.includes('400 10px "JetBrains Mono"') &&
    connectivitySource.includes("scheduleFontsReadyRedraw"),
  "CIFTI connectivity must redraw after all exact Inter and JetBrains canvas fonts resolve"
);

const packageJson = JSON.parse(packageSource);
check(
  packageJson.scripts["test:smoke"] ===
    "pnpm run test:smoke-safety && node scripts/run-mobile-smoke.mjs" &&
    packageJson.scripts["mobile:smoke"] === "pnpm run test:smoke",
  "All package smoke commands must run safety canaries before the wrapper"
);
check(
  smokeRunnerSource.includes("randomBytes(32)") &&
    smokeRunnerSource.includes("/v1/smoke/identity") &&
    smokeRunnerSource.includes("--strictPort") &&
    smokeRunnerSource.includes("waitForGuardedProcess"),
  "Smoke wrapper must issue a nonce, verify exclusive listeners, and monitor authorities"
);
check(
  mobileSmokeSource.includes("MOBILE_SMOKE_NONCE") &&
    mobileSmokeSource.includes("attachSmokePageGuard") &&
    mobileSmokeSource.includes("requireAuthenticatedMockShell") &&
    !mobileSmokeSource.includes("form.auth-form") &&
    !mobileSmokeSource.includes('waitUntil: "networkidle"'),
  "Direct smoke must require wrapper identity, refuse auth forms, and use the safe page adapter"
);
check(
  mockApiSource.includes('url.pathname === "/v1/smoke/identity"') &&
    mockApiSource.includes("smokeRunNonce"),
  "Mock API must expose the wrapper-bound smoke identity"
);

if (checkDist) {
  check(fs.existsSync(distRoot), "dist is missing; run the production build first");
  if (fs.existsSync(distRoot)) {
    const distFiles = walkFiles(distRoot);
    const distRelative = (file) => path.relative(distRoot, file).replaceAll(path.sep, "/");
    const interLike = distFiles.filter((file) => /intervariable/i.test(path.basename(file)));
    const interFontFiles = interLike.filter((file) => /\.(?:woff2?|ttf|otf)$/i.test(file));
    check(interFontFiles.length === 2, `Expected two emitted Inter font files, found ${interFontFiles.length}`);
    check(
      interFontFiles.every((file) => file.endsWith(".woff2")),
      "Production emitted a non-WOFF2 Inter font"
    );

    const emittedByHash = new Map(
      distFiles
        .filter((file) => file.endsWith(".woff2"))
        .map((file) => [digest(fs.readFileSync(file)), file])
    );
    const emittedExpected = expectedFonts.map((font) => ({
      ...font,
      emitted: emittedByHash.get(font.sha256),
    }));
    for (const font of emittedExpected) {
      check(Boolean(font.emitted), `Production build is missing exact ${font.style} Inter v4.1 bytes`);
      if (font.emitted) {
        check(
          fs.statSync(font.emitted).size === font.bytes,
          `Emitted ${font.style} Inter byte count changed`
        );
      }
    }
    const emittedTotal = emittedExpected.reduce(
      (total, font) => total + (font.emitted ? fs.statSync(font.emitted).size : 0),
      0
    );
    check(emittedTotal === 740_216, `Emitted Inter payload is ${emittedTotal} bytes; expected 740216`);

    const builtCssFiles = distFiles.filter((file) => file.endsWith(".css"));
    const cssWithInter = builtCssFiles.filter((file) =>
      fs.readFileSync(file, "utf8").includes("BisQue Inter Variable")
    );
    check(cssWithInter.length === 1, `Expected one emitted CSS asset with Inter faces, found ${cssWithInter.length}`);

    let normalFacePath = null;
    if (cssWithInter.length === 1) {
      const cssFile = cssWithInter[0];
      const css = fs.readFileSync(cssFile, "utf8");
      const builtFaces = cssBlocks(css, "@font-face").filter((block) =>
        block.includes("BisQue Inter Variable")
      );
      check(builtFaces.length === 2, `Expected two built Inter @font-face rules, found ${builtFaces.length}`);
      for (const font of expectedFonts) {
        const face = builtFaces.find((block) => new RegExp(`font-style:\\s*${font.style}`).test(block));
        check(Boolean(face), `Built CSS is missing ${font.style} Inter face`);
        if (!face) {
          continue;
        }
        check(/font-weight:\s*100 900/.test(face), `Built ${font.style} face lost weight 100–900`);
        check(/font-display:\s*swap/.test(face), `Built ${font.style} face lost font-display: swap`);
        check(/format\(["']?woff2["']?\)/.test(face), `Built ${font.style} face lost WOFF2 format`);
        const rawUrl = face.match(/url\((["']?)([^"')]+)\1\)/)?.[2];
        check(Boolean(rawUrl), `Built ${font.style} face is missing its URL`);
        if (rawUrl) {
          const cssRelative = distRelative(cssFile);
          const resolved = rawUrl.startsWith("/")
            ? rawUrl.slice(1)
            : path.posix.normalize(path.posix.join(path.posix.dirname(cssRelative), rawUrl));
          const expectedPath = emittedByHash.get(font.sha256);
          check(
            expectedPath && resolved === distRelative(expectedPath),
            `Built ${font.style} face URL does not resolve to the verified asset`
          );
          if (font.style === "normal") {
            normalFacePath = resolved;
          }
        }
      }
    }

    const allBuiltCss = builtCssFiles.map((file) => fs.readFileSync(file, "utf8")).join("\n");
    check(!/url\((["']?)https?:\/\//i.test(allBuiltCss), "Built CSS contains an external asset URL");
    check(!/@import\s+(?:url\()?["']?https?:\/\//i.test(allBuiltCss), "Built CSS contains an external import");
    const katexItalicFaces = cssBlocks(allBuiltCss, "@font-face").filter(
      (block) => /font-family:\s*KaTeX/.test(block) && /font-style:\s*italic/.test(block)
    );
    check(
      katexItalicFaces.length > 0,
      "Global synthesis policy requires genuine KaTeX italic faces in the production bundle"
    );

    const builtHtml = fs.readFileSync(path.join(distRoot, "index.html"), "utf8");
    const fontPreloads = [...builtHtml.matchAll(/<link\b[^>]*\bas=["']font["'][^>]*>/gi)];
    check(fontPreloads.length <= 1, `Expected at most one font preload, found ${fontPreloads.length}`);
    if (fontPreloads.length === 1) {
      const href = fontPreloads[0][0].match(/\bhref=["']([^"']+)["']/i)?.[1];
      check(Boolean(href), "Font preload is missing href");
      if (href) {
        check(href.replace(/^\//, "") === normalFacePath, "Font preload does not match normal-face CSS URL");
        const italicPath = emittedExpected.find(({ style }) => style === "italic")?.emitted;
        check(
          !italicPath || href.replace(/^\//, "") !== distRelative(italicPath),
          "Italic Inter must remain demand-loaded and must never be preloaded"
        );
      }
    }

    console.log("Verified emitted Inter inventory:");
    for (const font of emittedExpected) {
      if (font.emitted) {
        console.log(`- ${distRelative(font.emitted)}: ${font.bytes} bytes (${font.style})`);
      }
    }
    console.log(`- total: ${emittedTotal} bytes; font preloads: ${fontPreloads.length}`);
  }
}

if (failures.length > 0) {
  throw new Error(`Typography contract failed:\n- ${failures.join("\n- ")}`);
}

console.log(
  checkDist
    ? "Typography source and production-bundle contracts passed."
    : "Typography source contract passed."
);
