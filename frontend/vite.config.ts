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
        return deps.filter((dep) => !dep.includes("vendor-three"));
      },
    },
    rolldownOptions: {
      output: {
        manualChunks(id) {
          const normalizedId = id.replace(/\\/g, "/");
          if (!normalizedId.includes("node_modules")) {
            if (normalizedId.endsWith("/src/lib/api.ts")) {
              return "api-client";
            }
            if (normalizedId.includes("/src/features/resources/")) {
              return "resources-client";
            }
            if (normalizedId.includes("/src/features/admin/")) {
              return "admin-client";
            }
            if (normalizedId.includes("/src/features/training/")) {
              return "training-client";
            }
            if (normalizedId.includes("/src/features/chat/")) {
              return "chat-client";
            }
            return undefined;
          }
          if (
            normalizedId.includes("lucide-react") ||
            normalizedId.includes("use-stick-to-bottom") ||
            normalizedId.includes("tailwind-merge") ||
            normalizedId.includes("cmdk") ||
            normalizedId.includes("radix-ui") ||
            normalizedId.includes("@radix-ui") ||
            normalizedId.includes("@floating-ui")
          ) {
            return "vendor-ui";
          }
          if (normalizedId.includes("@react-three") || normalizedId.includes("/three/")) {
            return "vendor-three";
          }
          return undefined;
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
