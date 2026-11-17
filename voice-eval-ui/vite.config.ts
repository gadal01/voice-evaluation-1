import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// Dev: `npm run dev` serves the UI on :5173 and proxies to the two services, so the browser sees one origin
// (no CORS needed):  /api -> evaluation backend,  /auth and /ws -> the voice agent.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const backend = env.EVAL_BACKEND_URL || "http://127.0.0.1:8001";
  const agent = env.AGENT_PROXY_URL || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": backend,
        "/auth": agent,
        "/ws": { target: agent.replace(/^http/, "ws"), ws: true },
      },
    },
    test: { environment: "node" },
  };
});
