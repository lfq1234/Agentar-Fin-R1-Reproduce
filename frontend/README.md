# frontend

React + Vite + TypeScript 交互前端，用于演示复刻出的金融 Agent 能力。

## 运行

```bash
npm install
npm run dev        # http://localhost:5173
```

dev server 已配置把 `/api` 代理到后端 `http://localhost:8000`（见 `vite.config.ts`）。

## 构建

```bash
npm run build      # 产物在 dist/
npm run preview
```

## 接入后端

`src/App.tsx` 通过 `POST /api/v1/chat` 调用 [`backend`](../backend/) 的 Agent 运行时。
后端就绪后，把 `agent/runner.py` 与 `services/inference.py` 从 stub 接为真实推理即可。
