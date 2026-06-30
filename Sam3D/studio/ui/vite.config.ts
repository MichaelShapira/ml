import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
  // gaussian-splats-3d / transformers.js ship large prebuilt assets; don't choke optimizer.
  optimizeDeps: { exclude: ["@mkkellogg/gaussian-splats-3d", "@huggingface/transformers"] },
  build: { target: "es2022", chunkSizeWarningLimit: 4000 },
});
