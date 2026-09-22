import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The dev server and the API are separate processes; the proxy keeps the browser on one origin,
// so nothing in the app needs an absolute backend URL.
const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: backend, changeOrigin: false },
    },
  },
  build: {
    // Data URLs would hide an external reference from scripts/check-bundle.ts.
    assetsInlineLimit: 0,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.ts", "src/**/*.test.tsx", "scripts/**/*.test.ts"],
    // restoreMocks only restores vi.spyOn spies. Vitest 3 also cleared plain vi.fn()
    // call history as a side effect and Vitest 4 does not, so a test asserting a call
    // count saw every earlier test's calls too. Clear the history explicitly rather
    // than depending on what restoreMocks happens to do.
    restoreMocks: true,
    clearMocks: true,
  },
});
