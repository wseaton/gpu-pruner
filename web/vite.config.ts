import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies API and Prometheus-relay calls to a locally running
// gpu-pruner (`gpu-pruner --dashboard-addr 127.0.0.1:8080 ...`).
const GPU_PRUNER_URL = process.env.GPU_PRUNER_URL ?? "http://localhost:8080";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: GPU_PRUNER_URL, changeOrigin: true },
      "/prom": { target: GPU_PRUNER_URL, changeOrigin: true },
    },
  },
});
