import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// https://vite.dev/config/ and https://vitest.dev/config/
export default defineConfig({
  plugins: [react()],
  // Pin the dev server to a fixed port and FAIL rather than drift to 5174+ when
  // 5173 is busy. The S3 bucket CORS and Lambda Function URL CORS allowlist this
  // exact origin (http://localhost:5173); a silent port bump breaks the booth's
  // browser-direct calls with a CORS preflight failure.
  server: {
    port: 5173,
    strictPort: true,
  },
  // amazon-cognito-identity-js (via its `buffer` dependency) references the
  // Node `global` object, which does not exist in the browser. Map it to
  // `globalThis` so the SDK loads in both dev and the production bundle.
  define: {
    global: "globalThis",
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: true,
  },
});
