import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

const apiProxyTarget =
  typeof process.env.VITE_PROXY_API_TARGET === "string" &&
  process.env.VITE_PROXY_API_TARGET.trim().length > 0
    ? process.env.VITE_PROXY_API_TARGET.trim()
    : "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    manifest: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) {
            return undefined;
          }
          if (
            id.includes("react-markdown") ||
            id.includes("marked") ||
            id.includes("remark-") ||
            id.includes("rehype-") ||
            id.includes("/katex/")
          ) {
            return "vendor-markdown";
          }
          if (
            id.includes("lucide-react") ||
            id.includes("use-stick-to-bottom") ||
            id.includes("/cmdk/") ||
            id.includes("/radix-ui/")
          ) {
            return "vendor-ui";
          }
          if (id.includes("@react-three") || id.includes("/three/")) {
            return "vendor-three";
          }
          if (id.includes("/recharts/")) {
            return "vendor-charts";
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
    proxy: {
      "/v1": {
        target: apiProxyTarget,
        changeOrigin: false,
      },
      "/v3": {
        target: apiProxyTarget,
        changeOrigin: false,
      },
    },
  },
});
