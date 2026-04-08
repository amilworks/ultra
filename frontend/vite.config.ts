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
    },
  },
});
