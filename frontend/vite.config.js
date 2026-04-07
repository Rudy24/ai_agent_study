import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// 开发时把 /api、/health 转到本机 FastAPI，避免浏览器跨域
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
