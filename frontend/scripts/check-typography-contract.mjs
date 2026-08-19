import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const frontendRoot = path.resolve(import.meta.dirname, "..");
const sourceRoot = path.join(frontendRoot, "src");
const distRoot = path.join(frontendRoot, "dist");
const checkDist = process.argv.includes("--dist");
const failures = [];

// Inter. No longer the product face — it draws the wordmark and backstops the
// glyphs Ultra Sans lacks. Still pinned byte-for-byte.
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

// Ultra Sans, the product face: DM Sans plus generated tabular figures, Greek
// and reference-matched round capitals from Inter across the full two-axis
// design space, and a dormant slashed-zero feature — built by
// font-lab/build_ultra_tabular.py. That build is byte-reproducible, so pinning
// its digest here also pins tnum, the 104 grafted codepoints, the C/O/G/Q/Oslash
// redraw, and the zero feature. PROVENANCE.md pins the upstream digests.
const expectedUiFonts = [
  {
    file: "UltraSans-Variable.woff2",
    style: "normal",
    bytes: 126_824,
    sha256: "1794a295de6c0214c5d530d763668102bf23059cb3b75e248d4c80a1d6772758",
    upstreamSha256: "8cd08d97e89c24d0aa92edd2f0f4c8ee6195eee9b7c9f154865a58b02f0c1c0d",
  },
  {
    file: "UltraSans-Italic-Variable.woff2",
    style: "italic",
    bytes: 153_972,
    sha256: "753ab3be31d45a905d79fd25074864beca038eef37071b7bea1507354b677bfd",
    upstreamSha256: "22259c0cc8237221b80f44c76ba8d36e6bce3cda72779f5b2773643d499720ae",
  },
];

// Ultra Mono, the product monospace: DM Mono rebuilt from its Glyphs sources as
// a variable font (upstream ships three incompatible statics capped at 500),
// extended to a generated 600 continuing the family's own two-master weight
// vector, with Greek grafted from weight-matched JetBrains Mono instances.
// Built by font-lab/build_ultra_mono.py; byte-reproducible, digest = contract.
const expectedMonoFonts = [
  {
    file: "UltraMono-Variable.woff2",
    style: "normal",
    bytes: 40_244,
    sha256: "b3445619fbe749b4384a27af82b47bc159916a75e9a394fa4a9f23af41c6166d",
    upstreamSha256: "7e73628b3cd9f3a164eaf3109145a59e15a633f3a9d12a2509c2bb027fc25314",
  },
  {
    file: "UltraMono-Italic-Variable.woff2",
    style: "italic",
    bytes: 43_744,
    sha256: "16d1735d9fc8dcb6978182e2b46805a4988db9415a750b1784948e26e9985cb1",
    upstreamSha256: "a3ecd457114537a29921caca4d1a5eea926b031fa581d2d8363fb645fa77d4d5",
  },
];

// JetBrains Mono, vendored once as the mono coverage net (it also donated the
// Greek outlines above). Container-converted from the pinned google/fonts TTFs.
const expectedMonoCoverage = [
  {
    file: "JetBrainsMono-Variable.woff2",
    style: "normal",
    bytes: 71_736,
    sha256: "7b7f3419196f675a973d30cb70078749120caddea86c8547ebf54a8db2ca13af",
    upstreamSha256: "48715a42ec242c21e9f02692891e147d022299a52e48d5e413e1a942193ffeda",
  },
  {
    file: "JetBrainsMono-Italic-Variable.woff2",
    style: "italic",
    bytes: 76_452,
    sha256: "84ef9a6d8b91d130a05c2b5697756d7b8d0d72ce5aef9ddffdd66881fac8e9f1",
    upstreamSha256: "85ae2a5cd3f56baf1ce1c21a851322c58e3d8fbe8e8ad4a4d090a820dd7fe558",
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
check(
  sourceFontBinaries.length === 8,
  `Expected eight source font binaries (Ultra Sans x2, Inter x2, Ultra Mono x2, JetBrains coverage x2), found ${sourceFontBinaries.length}`
);
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

for (const font of expectedUiFonts) {
  const fontPath = path.join(sourceFontDir, font.file);
  check(fs.existsSync(fontPath), `Missing source font: ${font.file}`);
  if (fs.existsSync(fontPath)) {
    const bytes = fs.readFileSync(fontPath);
    check(bytes.length === font.bytes, `${font.file} has ${bytes.length} bytes; expected ${font.bytes}`);
    check(digest(bytes) === font.sha256, `${font.file} SHA-256 does not match the vendored conversion`);
  }
  for (const provenanceValue of [
    font.file,
    String(font.bytes),
    font.sha256,
    // The upstream TTF digest is what makes the WOFF2 conversion auditable.
    font.upstreamSha256,
    "`wght` 100–1000",
    "`opsz` 9–40",
  ]) {
    check(provenance.includes(provenanceValue), `Provenance is missing ${provenanceValue}`);
  }
}

// What the derivative is, why it is licensed to exist, and what it still lacks
// must stay written down rather than be rediscovered from the binaries.
check(
  /no Reserved Font Name/i.test(provenance),
  "Provenance must record that DM Sans declares no Reserved Font Name"
);
check(
  /build_ultra_tabular\.py/.test(provenance) && /310\/1000em/.test(provenance),
  "Provenance must name the build script and the digit spread it exists to fix"
);
check(
  /byte-reproducible/.test(provenance),
  "Provenance must record that the derivative build is byte-reproducible, since its digest is pinned"
);
check(
  /104 Greek codepoints/.test(provenance) && /12 master locations/.test(provenance),
  "Provenance must record the Greek graft and the master grid it is exact at"
);
check(
  /slashed[- ]zero/i.test(provenance) && /dormant/i.test(provenance),
  "Provenance must record the slashed-zero feature and that it ships dormant"
);
check(
  /HVAR/.test(provenance) && /phantom/.test(provenance),
  "Provenance must record the HVAR drop and the phantom-advance rebuild"
);
check(
  /name ID 0/.test(provenance) && /13\/14/.test(provenance),
  "Provenance must record that upstream copyright and license name IDs travel untouched"
);

for (const font of [...expectedMonoFonts, ...expectedMonoCoverage]) {
  const fontPath = path.join(sourceFontDir, font.file);
  check(fs.existsSync(fontPath), `Missing source font: ${font.file}`);
  if (fs.existsSync(fontPath)) {
    const bytes = fs.readFileSync(fontPath);
    check(bytes.length === font.bytes, `${font.file} has ${bytes.length} bytes; expected ${font.bytes}`);
    check(digest(bytes) === font.sha256, `${font.file} SHA-256 does not match the vendored build`);
  }
  for (const provenanceValue of [font.file, String(font.bytes), font.sha256, font.upstreamSha256]) {
    check(provenance.includes(provenanceValue), `Provenance is missing ${provenanceValue}`);
  }
}

// Mono derivative facts that must stay written down:
check(
  /build_ultra_mono\.py/.test(provenance) && /wght` 300–600/.test(provenance),
  "Provenance must name the mono build script and its extended weight axis"
);
check(
  /two drawn masters/.test(provenance) && /stem 70/.test(provenance),
  "Provenance must record that the 600 continues DM Mono's own two-master weight vector"
);
check(
  /69% of its (500 )?area/.test(provenance),
  "Provenance must record the measured worst-counter survival at the generated 600"
);
check(
  /grid outranks/.test(provenance),
  "Provenance must record why the mono coverage face carries NO size-adjust (grid > optical match)"
);
check(
  /0\.169/.test(provenance) && /0\.165/.test(provenance),
  "Provenance must record the nominal-weight decision with the measured stem/x-height ratios"
);

// Licenses for the two mono sources, byte-pinned like the others.
for (const [file, bytes, sha] of [
  ["OFL-1.1-DMMono.txt", 4_484, "2bada5ea45c3c63b7f1ea1f88ce9672c9e4f0c42b2c3b7378949084fe55a3066"],
  ["OFL-1.1-JetBrainsMono.txt", 4_399, "b2fe5e8987594e9ffd1d2ca52a2f5d73eb8335243893c5d6254b5ad69269591d"],
]) {
  const p = path.join(sourceFontDir, file);
  check(fs.existsSync(p), `Missing ${file}`);
  if (fs.existsSync(p)) {
    const b = fs.readFileSync(p);
    check(b.length === bytes, `${file} byte count changed`);
    check(digest(b) === sha, `${file} is not the verbatim upstream notice`);
  }
}

// The mono generator is part of the shipped binary's provenance, like the
// tabular builder below.
const monoBuilder = path.join(frontendRoot, "font-lab", "build_ultra_mono.py");
check(
  fs.existsSync(monoBuilder),
  "font-lab/build_ultra_mono.py is missing; it builds the shipped Ultra Mono binaries"
);
if (fs.existsSync(monoBuilder)) {
  const builder = fs.readFileSync(monoBuilder, "utf8");
  check(
    builder.includes("recalcTimestamp"),
    "Mono build must pin timestamps, or its digest stops being reproducible"
  );
  check(
    builder.includes("contour_signed_areas") && builder.includes("scanline_stem"),
    "Mono build must keep its counter-collapse and stem-progression checks"
  );
}

// The shipped product face is generated, not downloaded, so the generator is
// part of its provenance: without it in the tree, the binary cannot be
// regenerated or audited and PROVENANCE.md points at nothing.
const tabularBuilder = path.join(frontendRoot, "font-lab", "build_ultra_tabular.py");
check(
  fs.existsSync(tabularBuilder),
  "font-lab/build_ultra_tabular.py is missing; it builds the shipped Ultra Sans binaries"
);
if (fs.existsSync(tabularBuilder)) {
  const builder = fs.readFileSync(tabularBuilder, "utf8");
  check(
    builder.includes("recalcTimestamp=False"),
    "Tabular build must pin head.modified, or its digest stops being reproducible"
  );
  check(
    builder.includes("assert_zero_is_widest"),
    "Tabular build must keep re-checking the zero-is-widest invariant it relies on"
  );
  check(
    builder.includes("graft_greek") && builder.includes("VariationModel"),
    "Sans build must keep the model-derived Greek graft"
  );
  check(
    builder.includes("drop_hvar_after_proof") && builder.includes("add_slashed_zero"),
    "Sans build must keep the HVAR phantom rebuild and the slashed-zero generator"
  );
  check(
    builder.includes("redraw_round_capitals(") &&
      builder.includes("_solve_inter_round_weight") &&
      builder.includes('ROUND_CAP_CHARS = ("C", "O", "G", "Q", "Ø")'),
    "Sans build must keep the reference-matched C/O/G/Q/Oslash redraw"
  );
}

const dmLicensePath = path.join(sourceFontDir, "OFL-1.1-DMSans.txt");
check(fs.existsSync(dmLicensePath), "Missing DM Sans OFL-1.1 license");
if (fs.existsSync(dmLicensePath)) {
  const dmLicenseBytes = fs.readFileSync(dmLicensePath);
  check(dmLicenseBytes.length === 4_482, "DM Sans OFL-1.1 license byte count changed");
  check(
    digest(dmLicenseBytes) === "9af36190332437f5ecd09974de43c1f7c77a310a996cdd8ceb25628b458840e1",
    "DM Sans OFL-1.1 license is not the verbatim upstream notice"
  );
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
const uiFaces = cssBlocks(typographyCss, "@font-face").filter((block) =>
  block.includes("BisQue Ultra Sans")
);
check(uiFaces.length === 2, `Expected two Ultra Sans @font-face rules, found ${uiFaces.length}`);
for (const font of expectedUiFonts) {
  const face = uiFaces.find((block) => new RegExp(`font-style:\\s*${font.style}`).test(block));
  check(Boolean(face), `Missing ${font.style} Ultra Sans face`);
  if (face) {
    check(face.includes(font.file), `${font.style} Ultra Sans face does not reference ${font.file}`);
    check(/font-weight:\s*100 1000\s*;/.test(face), `${font.style} Ultra Sans face must expose weight 100–1000`);
    check(/font-display:\s*swap\s*;/.test(face), `${font.style} Ultra Sans face must use font-display: swap`);
    check(/format\(["']woff2["']\)/.test(face), `${font.style} Ultra Sans face must declare WOFF2`);
    check(!/size-adjust/.test(face), `${font.style} Ultra Sans face must render at its own scale`);
  }
}

// The coverage backstop. Without it, every Greek letter Ultra renders drops to
// the platform font at a visibly different size.
const coverageFaces = cssBlocks(typographyCss, "@font-face").filter((block) =>
  block.includes("BisQue Inter Coverage")
);
check(coverageFaces.length === 2, `Expected two Inter coverage faces, found ${coverageFaces.length}`);
for (const face of coverageFaces) {
  check(
    /size-adjust:\s*96\.4%\s*;/.test(face),
    "Coverage face must carry size-adjust: 96.4% (DM Sans x-height 526.0 over Inter's 545.9)"
  );
  check(
    !/unicode-range/.test(face),
    "Coverage face must stay unscoped; a unicode-range would drop non-Greek gaps to the platform font"
  );
}
check(
  /--font-sans:\s*"BisQue Ultra Sans",\s*"BisQue Inter Coverage",/.test(typographyCss) &&
    /--font-reading:\s*"BisQue Ultra Sans",\s*"BisQue Inter Coverage",/.test(typographyCss),
  "UI and reading stacks must lead with Ultra Sans and fall back to the Inter coverage face before system-ui"
);
check(
  /--font-brand:\s*"BisQue Inter Variable",/.test(typographyCss),
  "Wordmark must keep unadjusted Inter via --font-brand"
);
check(
  /\.brand-wordmark\s*\{[^}]*font-family:\s*var\(--font-brand\)/s.test(typographyCss),
  "Wordmark must consume --font-brand, not the Ultra Sans UI stack"
);

// Ultra Mono faces: variable 300–600 at its own scale.
const monoFaces = cssBlocks(typographyCss, "@font-face").filter((block) =>
  block.includes("BisQue Ultra Mono")
);
check(monoFaces.length === 2, `Expected two Ultra Mono @font-face rules, found ${monoFaces.length}`);
for (const font of expectedMonoFonts) {
  const face = monoFaces.find((block) => new RegExp(`font-style:\\s*${font.style}`).test(block));
  check(Boolean(face), `Missing ${font.style} Ultra Mono face`);
  if (face) {
    check(face.includes(font.file), `${font.style} Ultra Mono face does not reference ${font.file}`);
    check(/font-weight:\s*300 600\s*;/.test(face), `${font.style} Ultra Mono face must expose weight 300–600`);
    check(!/size-adjust/.test(face), `${font.style} Ultra Mono face must render at its own scale`);
  }
}

// Mono coverage faces: JetBrains at 100% — NO size-adjust, deliberately and in
// contrast to the sans coverage face. Both fonts share the 600/1000em cell, so
// unscaled fallback glyphs stay on the mono grid; scaling to match x-height
// would put every substituted glyph off-grid.
const monoCoverageFaces = cssBlocks(typographyCss, "@font-face").filter((block) =>
  block.includes("BisQue Mono Coverage")
);
check(monoCoverageFaces.length === 2, `Expected two mono coverage faces, found ${monoCoverageFaces.length}`);
for (const face of monoCoverageFaces) {
  check(!/size-adjust/.test(face), "Mono coverage must NOT size-adjust — fallback glyphs must stay on the 600-unit grid");
  check(!/unicode-range/.test(face), "Mono coverage must stay unscoped");
  check(/font-weight:\s*100 800\s*;/.test(face), "Mono coverage must expose JetBrains' full 100–800 range");
}

// One mono voice: the token exists, leads with Ultra Mono, and no rule anywhere
// re-declares a JetBrains or ad-hoc system stack.
check(
  /--font-mono:\s*"BisQue Ultra Mono",\s*"BisQue Mono Coverage",\s*ui-monospace,/.test(typographyCss),
  "--font-mono must lead with Ultra Mono, then the coverage face, then system monos"
);
const nonTypographyCssFiles = walkFiles(sourceRoot).filter(
  (file) => file.endsWith(".css") && path.basename(file) !== "typography.css"
);
for (const cssFile of nonTypographyCssFiles) {
  const css = fs.readFileSync(cssFile, "utf8");
  check(
    !css.includes('"JetBrains Mono"'),
    `${path.relative(frontendRoot, cssFile)} re-declares a JetBrains Mono stack; use var(--font-mono)`
  );
  for (const match of css.matchAll(/font-family\s*:\s*([^;]*monospace[^;]*);/gi)) {
    check(
      match[1].includes("var(--font-mono"),
      `${path.relative(frontendRoot, cssFile)} has an ad-hoc mono stack "${match[1].slice(0, 60)}"; use var(--font-mono)`
    );
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
// The mono migrated from 14 per-weight @fontsource imports (28 emitted files,
// 508K) to two vendored variable faces. Nothing may quietly reintroduce it.
check(
  !mainSource.includes("@fontsource/jetbrains-mono"),
  "Production entrypoint still imports @fontsource/jetbrains-mono"
);
check(
  !packageSource.includes("@fontsource/jetbrains-mono"),
  "package.json still depends on @fontsource/jetbrains-mono"
);
check(
  !lockSource.includes("jetbrains-mono"),
  "pnpm lockfile still contains jetbrains-mono"
);
// Scientific code comments require a genuine mono italic; Ultra Mono ships one
// (variable 300–600) and .tv-c must keep consuming it as true 400 italic.
check(
  /\.tv-c\s*\{[^}]*font-weight:\s*400;[^}]*font-style:\s*italic;/s.test(stylesCss),
  "Scientific code-comment syntax must remain genuine 400 italic"
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
  // At opsz 15/16, Ultra Sans w430 has a 0.0888em lowercase stem over a
  // 0.5027/0.5020em x-height. W440 expands the stem to 0.0904em, which adds
  // unnecessary ink in Night while preserving no useful hierarchy.
  ["body", "430"],
  ["reading-body", "430"],
  // The welcome question is deliberately lighter than the reading voice:
  // w350 remains calm and legible where w300 became fragile on the Day ground.
  ["invitation", "350"],
  ["nav", "500"],
  ["action", "500"],
  ["data", "500"],
  ["label", "600"],
  ["panel-heading", "600"],
  ["page-heading", "600"],
  ["reading-heading", "600"],
  // Retired Inter opsz24/w670 measured lowercase stem/x-height 0.262550. Ultra
  // Sans pinned at opsz13 reaches 0.26254 at w689.18, rounded to 690. It still
  // outranks reading-heading while matching the intended optical mass.
  ["strong", "690"],
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
  [
    /--reading-measure:\s*42\.1875rem;/,
    "Response prose measure must preserve the measured 45-em calibration at 15px",
  ],
  [/--user-chat-width:\s*49rem;/, "Desktop chat reading measure must remain 49rem"],
  [/@media \(max-width: 640px\)[\s\S]*--line-height-reading:\s*1\.68;/, "Phone reading line-height must remain 1.68"],
  [/@media \(max-width: 640px\)[\s\S]*--font-size-body:\s*1rem;/, "Phone body must remain 16px"],
  [/@media \(max-width: 640px\)[\s\S]*--font-size-reading:\s*1rem;/, "Phone reading must remain 16px"],
  [/\.pk-prompt-input-textarea\s*\{[^}]*font:\s*inherit;/s, "Composer must inherit the 16px phone font"],
  // Mono surfaces must pin their weight. Static mono faces + CSS rounding turn
  // an inherited body weight into a silent grade jump (430 -> 500); these two
  // registers are the reading surfaces where that reads as shouting.
  [/\.pk-inline-code\s*\{[^}]*font-weight:\s*var\(--font-weight-mono\);/s, "Inline code must pin var(--font-weight-mono)"],
  [/\.pk-code-render :where\(pre, code\)\s*\{[^}]*font-weight:\s*var\(--font-weight-mono\);/s, "Code blocks must pin var(--font-weight-mono)"],
]) {
  check(pattern.test(stylesCss), message);
}

for (const [token, value] of [
  ["display-regular", "-0.01em"],
  ["display-strong", "-0.018em"],
  ["reading-h1", "-0.016em"],
  ["reading-h2", "-0.011em"],
  ["reading-h3", "-0.007em"],
  ["reading-small", "-0.003em"],
]) {
  check(
    new RegExp(`--tracking-${token}:\\s*${value.replace(".", "\\.")};`).test(stylesCss),
    `Root Ultra Sans tracking role --tracking-${token} must be ${value}`
  );
}
check(
  /\.app-composer-textarea\s*\{[^}]*letter-spacing:\s*0;/s.test(stylesCss),
  "Composer must use Ultra Sans native tracking at its rendered 15px size"
);
check(
  /\.blank-chat-welcome-hero\s*\{[^}]*font-weight:\s*var\(--font-weight-invitation\);/s.test(stylesCss) &&
    /\.mobile-chat-hero-title\s*\{[^}]*font-weight:\s*var\(--font-weight-invitation\);/s.test(stylesCss),
  "Desktop and mobile New Chat invitations must consume the light invitation role"
);

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
  ["context-light", "#424547", "#f2f3f3"],
  ["emphasis-light", "#171b1d", "#f2f3f3"],
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
  carpetSource.includes('600 11px "BisQue Ultra Sans"') &&
    carpetSource.includes('400 11px "BisQue Ultra Mono"') &&
    carpetSource.includes('400 10px "BisQue Ultra Mono"') &&
    carpetSource.includes("scheduleFontsReadyRedraw"),
  "CIFTI carpet must redraw after all exact Ultra Sans and Ultra Mono canvas fonts resolve"
);
check(
  connectivitySource.includes('600 10px "BisQue Ultra Sans"') &&
    connectivitySource.includes('400 10px "BisQue Ultra Mono"') &&
    connectivitySource.includes("scheduleFontsReadyRedraw"),
  "CIFTI connectivity must redraw after all exact Ultra Sans and Ultra Mono canvas fonts resolve"
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

    // Ultra Sans is the product face; verify the exact bytes reached dist too, or a
    // dropped asset would only surface as the whole UI silently rendering in the
    // Inter coverage face.
    const emittedUi = expectedUiFonts.map((font) => ({
      ...font,
      emitted: emittedByHash.get(font.sha256),
    }));
    for (const font of emittedUi) {
      check(Boolean(font.emitted), `Production build is missing exact ${font.style} Ultra Sans bytes`);
      if (font.emitted) {
        check(
          fs.statSync(font.emitted).size === font.bytes,
          `Emitted ${font.style} Ultra Sans byte count changed`
        );
      }
    }
    const emittedUiTotal = emittedUi.reduce(
      (total, font) => total + (font.emitted ? fs.statSync(font.emitted).size : 0),
      0
    );
    check(emittedUiTotal === 280_796, `Emitted Ultra Sans payload is ${emittedUiTotal} bytes; expected 280796`);

    // Mono: Ultra Mono + its coverage net, by digest; and the fontsource
    // static fleet must be gone from dist entirely.
    for (const [fonts, label, expectedTotal] of [
      [expectedMonoFonts, "Ultra Mono", 83_988],
      [expectedMonoCoverage, "mono coverage", 148_188],
    ]) {
      let total = 0;
      for (const font of fonts) {
        const emitted = emittedByHash.get(font.sha256);
        check(Boolean(emitted), `Production build is missing exact ${font.style} ${label} bytes`);
        if (emitted) {
          total += fs.statSync(emitted).size;
        }
      }
      check(total === expectedTotal, `Emitted ${label} payload is ${total} bytes; expected ${expectedTotal}`);
    }
    const fontsourceLeftovers = distFiles.filter((file) =>
      /jetbrains-mono-latin/.test(path.basename(file))
    );
    check(
      fontsourceLeftovers.length === 0,
      `dist still contains ${fontsourceLeftovers.length} fontsource JetBrains files`
    );

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

    const cssWithUi = builtCssFiles.filter((file) =>
      fs.readFileSync(file, "utf8").includes("BisQue Ultra Sans")
    );
    check(cssWithUi.length === 1, `Expected one emitted CSS asset with Ultra Sans faces, found ${cssWithUi.length}`);
    if (cssWithUi.length === 1) {
      const css = fs.readFileSync(cssWithUi[0], "utf8");
      const builtUiFaces = cssBlocks(css, "@font-face").filter((block) =>
        block.includes("BisQue Ultra Sans")
      );
      check(builtUiFaces.length === 2, `Expected two built Ultra Sans @font-face rules, found ${builtUiFaces.length}`);
      check(
        builtUiFaces.every((block) => /font-weight:\s*100 1000/.test(block)),
        "Built Ultra Sans faces lost weight 100–1000"
      );
      // Minifiers rewrite percentages; if this is ever dropped or normalised the
      // Greek backstop silently starts rendering a size too large.
      const builtCoverage = cssBlocks(css, "@font-face").filter((block) =>
        block.includes("BisQue Inter Coverage")
      );
      check(builtCoverage.length === 2, `Expected two built coverage faces, found ${builtCoverage.length}`);
      check(
        builtCoverage.every((block) => /size-adjust:\s*96\.4%/.test(block)),
        "Built coverage face lost size-adjust: 96.4%"
      );
      const builtMono = cssBlocks(css, "@font-face").filter((block) =>
        block.includes("BisQue Ultra Mono")
      );
      check(builtMono.length === 2, `Expected two built Ultra Mono faces, found ${builtMono.length}`);
      check(
        builtMono.every((block) => /font-weight:\s*300 600/.test(block)),
        "Built Ultra Mono faces lost weight 300–600"
      );
      const builtMonoCoverage = cssBlocks(css, "@font-face").filter((block) =>
        block.includes("BisQue Mono Coverage")
      );
      check(builtMonoCoverage.length === 2, `Expected two built mono coverage faces, found ${builtMonoCoverage.length}`);
      check(
        builtMonoCoverage.every((block) => !/size-adjust/.test(block)),
        "Built mono coverage face gained a size-adjust — fallback glyphs would leave the 600-unit grid"
      );
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
