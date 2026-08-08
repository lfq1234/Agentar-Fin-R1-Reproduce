# backend

Python + FastAPI 服务与 Agent 运行时。

## 结构

```
app/
├── main.py          # FastAPI 入口（/api/health, /api/v1/chat, /api/v1/analyze）
├── routes/chat.py   # HTTP 路由（传统 REST，全 async）
├── agent/           # 02 多智能体框架（仅调用 run()）
├── db/
│   ├── schema/sqlite/main.sql   # 可执行生产建表 DDL（users/conversations，单表收口）
│   ├── history/      # 07 会话历史：收口进 conversations.data 的存储/检索/迁移
│   └── models/      # SQLModel 表模型 + 请求/响应校验模型（同处一包）
│       ├── connection.py        # 引擎/会话/get_db/init_db（enabled 门控）
│       ├── user.py              # User 表
│       ├── conversation.py      # Conversation 表 + ChatRequest/ChatResponse
│       └── message.py           # AnalyzeRequest/AnalyzeResponse 契约（Message 表已并入 conversations.data）
└── services/        # chat_service / analyze_service（消费 agent.run，落库）
tests/               # pytest 测试
```

## 运行

```bash
uv pip install fastapi uvicorn pydantic sqlmodel httpx pyyaml openai agentscope
uvicorn app.main:app --reload --port 8000
# 或
python -m uvicorn app.main:app --reload --port 8000
```

## 测试

```bash
pytest
```

## 说明

`agent/runner.py` 与 `services/inference.py` 为早期 stub，已删除；当前 `/api/v1/chat`、`/api/v1/analyze`
由 `app/services/` 调用 02 的 `app.agent.run()` 实现，请求/响应校验模型合并在 `app/db/models/`
（非表 SQLModel，底层即 Pydantic）。
