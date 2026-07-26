# backend

Python + FastAPI 服务与 Agent 运行时。

## 结构

```
app/
├── main.py          # FastAPI 入口（/health, /v1/chat, /v1/analyze）
├── routes/chat.py   # HTTP 路由
├── agent/runner.py  # Agent 运行时：意图/槽位/工具规划/表达生成
├── models/schemas.py# 请求/响应 schema
└── services/inference.py  # 模型推理（加载 training 产出的检查点）
tests/               # pytest 测试
```

## 运行

```bash
uv pip install fastapi uvicorn pydantic httpx
uvicorn app.main:app --reload --port 8000
# 或
python -m uvicorn app.main:app --reload --port 8000
```

## 测试

```bash
pytest
```

## 说明

当前 `agent/runner.py` 与 `services/inference.py` 为 stub，待 `agentar-fin-r1/training`
产出模型后接入真实推理与工具调用。
