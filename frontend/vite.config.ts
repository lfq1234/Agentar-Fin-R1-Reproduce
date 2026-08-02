import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 前端 dev server 把 /api 代理到后端（:8000），避免跨域。
// 路径与后端一致（统一带 /api 前缀）：前端 /api/v1/chat → 代理 → 后端 /api/v1/chat。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
