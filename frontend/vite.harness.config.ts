import path from "node:path";
import { defineConfig } from "vite";

/**
 * Builds ONLY the dev scene3d harness (scene3d-harness.html), separately from the app,
 * so verifying the wire format on a real GPU never adds a page or a byte to the product
 * bundle. Driven by scripts/verify-scene3d.mjs, which serves this output plus the derived
 * fixture over an ephemeral port and drives it with Playwright.
 */
export default defineConfig({
  root: __dirname,
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: process.env.HARNESS_OUT_DIR ?? path.resolve(__dirname, "../.tmp/scene3d-harness"),
    emptyOutDir: true,
    // The harness is a debugging tool; readable output beats small output.
    minify: false,
    rollupOptions: {
      input: path.resolve(__dirname, "scene3d-harness.html"),
    },
  },
});
