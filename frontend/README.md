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

`src/api/client.ts` 通过 `/api/v1/chat`、`/api/v1/analyze`、`/api/health` 调用 [`backend`](../backend/) 的 Agent 运行时（`app/routes/chat.py`）。后端就绪后，把 `app/agent/system.py` 的 `run()` 从 stub 接为真实推理即可。

## 接口契约

- `GET  /api/health`
- `POST /api/v1/chat` → `{ message, scene?, user_id?, conversation_id? }` → `{ scene?, conversation_id?, reply, compliance_notes[], risk_flags[] }`
- `POST /api/v1/analyze` → `{ message, scene? }` → `{ scene?, intent?, slots{}, tool_plan[], expression }`

续聊 / 落库前提：后端 `db.enabled` 默认 `false`，且 `chat` 仅在「db 可用 且 传入 `user_id`」时返回 `conversation_id`。前端默认携带 `user_id=1`。
