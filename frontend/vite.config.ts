import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

const apiProxyTarget =
  typeof process.env.VITE_PROXY_API_TARGET === "string" &&
  process.env.VITE_PROXY_API_TARGET.trim().length > 0
    ? process.env.VITE_PROXY_API_TARGET.trim()
    : "http://localhost:8000";

const apiProxy = {
  "/v1": {
    target: apiProxyTarget,
    changeOrigin: false,
  },
  "/v2": {
    target: apiProxyTarget,
    changeOrigin: false,
  },
  "/v3": {
    target: apiProxyTarget,
    changeOrigin: false,
  },
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    manifest: true,
    modulePreload: {
      resolveDependencies(_url, deps, { hostType }) {
        if (hostType !== "html") {
          return deps;
        }
        return deps.filter(
          (dep) => !dep.includes("vendor-three") && !dep.includes("vendor-spark")
        );
      },
    },
    rolldownOptions: {
      output: {
        // Chunking is expressed entirely as advancedChunks groups rather than
        // manualChunks. The two are mutually exclusive — supplying groups disables
        // the manualChunks compat layer — and groups are what we actually need:
        //
        // three.js is imported by BOTH the volume viewer and Spark's splat runtime.
        // For such a shared module, a manualChunks name is only a hint. Rolldown
        // hoisted three's core into vendor-spark and left vendor-three a 32 KB addons
        // stub that imported it, so opening any NIfTI volume pulled the whole 5.5 MB
        // splat runtime. An advancedChunks group is a hard split, and the higher
        // priority on vendor-three is what keeps three's core out of Spark's chunk.
        // check-bundle-budgets.mjs asserts the isolation so it cannot silently regress.
        advancedChunks: {
          groups: [
            {
              name: "vendor-three",
              priority: 100,
              test: (id) => /[\\/]three[\\/]/.test(id.replace(/\\/g, "/")),
            },
            {
              name: "vendor-spark",
              priority: 90,
              test: (id) => id.replace(/\\/g, "/").includes("/@sparkjsdev/"),
            },
            {
              name: "vendor-ui",
              priority: 80,
              test: (id) => {
                const normalizedId = id.replace(/\\/g, "/");
                if (!normalizedId.includes("node_modules")) {
                  return false;
                }
                // cmdk is only reached through lazy components (ComposerSlashMenu,
                // Hdf5Navigator), so it is intentionally NOT forced into the eager
                // vendor-ui chunk — Rolldown co-locates it with those lazy chunks.
                return (
                  normalizedId.includes("lucide-react") ||
                  normalizedId.includes("use-stick-to-bottom") ||
                  normalizedId.includes("tailwind-merge") ||
                  normalizedId.includes("radix-ui") ||
                  normalizedId.includes("@radix-ui") ||
                  normalizedId.includes("@floating-ui")
                );
              },
            },
            {
              name: "api-client",
              priority: 70,
              test: (id) => {
                const normalizedId = id.replace(/\\/g, "/");
                return (
                  !normalizedId.includes("node_modules") &&
                  normalizedId.endsWith("/src/lib/api.ts")
                );
              },
            },
            ...["resources", "admin", "training", "chat"].map((feature) => ({
              name: `${feature}-client`,
              priority: 60,
              test: (id) => {
                const normalizedId = id.replace(/\\/g, "/");
                return (
                  !normalizedId.includes("node_modules") &&
                  normalizedId.includes(`/src/features/${feature}/`)
                );
              },
            })),
          ],
        },
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "localhost",
    port: 5173,
    proxy: apiProxy,
  },
  preview: {
    proxy: apiProxy,
  },
});
