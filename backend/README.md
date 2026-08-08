# backend

Python + FastAPI 服务与 Agent 运行时。基于 AgentScope 多智能体框架，Qwen3-0.6B 本地直载。

## 目录结构

```
backend/
├── config/
│   ├── config.yaml              # 运行时配置（model/kb/db/history/security）
│   └── config.example.yaml      # 配置模板（api 模式示例）
├── .env / .env.example          # 环境变量（AUTH_SECRET_KEY 等）
├── app/
│   ├── main.py                  # FastAPI 入口：lifespan + CORS + 路由挂载
│   ├── config/__init__.py       # YAML 配置加载器（${ENV_VAR} 占位符替换）
│   ├── core/
│   │   └── security.py          # JWT (HS256) + Argon2/Bcrypt 密码哈希
│   ├── crud/
│   │   └── user.py              # 用户 CRUD + 认证
│   ├── db/
│   │   ├── models/              # SQLModel ORM（users/conversations）
│   │   │   ├── connection.py    # 异步引擎/会话/get_db（enabled 门控）
│   │   │   ├── user.py          # User 表 + API 契约
│   │   │   ├── conversation.py  # Conversation 表 + ChatRequest/ChatResponse
│   │   │   └── message.py       # AnalyzeRequest/AnalyzeResponse
│   │   ├── migrate.py           # 轻量迁移（幂等 ALTER）
│   │   ├── history/             # 07 会话历史（单表 conversations.data）
│   │   │   ├── store.py         # SQLite 存储 + 权限仲裁 + 降级
│   │   │   ├── hooks.py         # 无侵入采集（install_history_tracing）
│   │   │   ├── search.py        # 语义/全文搜索
│   │   │   ├── export.py        # 导出（md/json）
│   │   │   └── redact.py        # PII 脱敏
│   │   ├── knowledge/           # 06 公共知识库
│   │   │   ├── store.py         # DuckDB 向量检索 + SQLite 存储
│   │   │   ├── chunking.py      # 文本分块 + BM25 + Passage
│   │   │   └── local_embed.py   # 进程内嵌入直载（分批处理防 OOM）
│   │   └── personal_docs/       # 08 个人文档与知识图谱
│   │       ├── store.py         # SQLite 存储（四表）
│   │       ├── schemas.py       # Pydantic 响应契约
│   │       ├── parser.py        # 文档解析（txt/pdf/docx）
│   │       ├── vector.py        # DuckDB 向量索引
│   │       └── graph_extract.py # 知识图谱抽取（启发式 + LLM）
│   ├── model/                   # 01 模型接口抽象层
│   │   ├── base.py              # ModelInterface / EmbedderInterface
│   │   ├── factory.py           # get_model() / get_embedder()
│   │   ├── api/                 # OpenAI 兼容 API 模式
│   │   └── local/               # 进程内 transformers 直载
│   │       ├── transformer_local.py  # Qwen3 推理
│   │       └── embed_local.py        # 嵌入（共享 LLM 权重）
│   ├── agent/                   # 02 多智能体框架
│   │   ├── system.py            # run() 同步 + run_stream() SSE 流式
│   │   ├── agents.py            # 9 个智能体定义（Coordinator + 5专家 + Direct + RAG + Review + Risk）
│   │   ├── board.py             # ExpertBoard 专家互询/圆桌会商编排
│   │   ├── model_bridge.py      # AgentarModel：AgentScope ↔ 01 模型层桥接
│   │   ├── tools.py             # 金融工具注册表
│   │   ├── rag_scope.py         # ContextVar 检索作用域
│   │   └── prompts/             # 10 个 Prompt 模板（txt）
│   ├── routes/                  # HTTP 路由
│   │   ├── chat.py              # /api/v1/chat + /api/v1/chat/stream(SSE)
│   │   ├── auth.py              # /api/v1/auth/*
│   │   ├── history.py           # /api/v1/history/*
│   │   ├── documents.py         # /api/v1/documents/* + /knowledge-graph
│   │   └── deps.py              # CurrentUser / SessionDep 依赖注入
│   └── services/                # 业务服务层
│       ├── chat_service.py      # chat() 编排 + 落库
│       ├── analyze_service.py   # 结构化分析
│       └── documents_service.py # 文档 ingest 流水线 + RAG 检索
├── tests/                       # pytest 测试（15 个文件）
└── tools/                       # 诊断工具
```

## 运行

```bash
cd backend
pip install fastapi uvicorn pydantic sqlmodel httpx pyyaml openai agentscope aiosqlite duckdb python-multipart pypdf pwdlib[argon2,bcrypt] PyJWT

# local 模式（Qwen3-0.6B 进程内直载）
uvicorn app.main:app --reload --port 8000

# api 模式（需 OPENAI_API_KEY）
# 复制 config/config.example.yaml → config/config.yaml，改 model.mode=api
```

## 配置要点

`config/config.yaml` 关键字段：

| 段 | 字段 | 说明 |
| --- | --- | --- |
| `model.mode` | `local` / `api` | local=进程内直载 Qwen3，api=OpenAI 兼容端点 |
| `model.local.model_path` | 路径 | Qwen3 权重目录（可用 `MODEL_PATH` env 覆盖） |
| `db.enabled` | bool | 启用 SQLite 落库（会话持久化/用户系统） |
| `kb.engine` | `duckdb` / `sqlite` | 知识库引擎（默认 duckdb 向量检索） |
| `kb.local_embed.model_path` | 路径 | 嵌入模型权重（默认复用 LLM 的 Qwen3-0.6B） |
| `security.secret_key` | `\${AUTH_SECRET_KEY}` | JWT 密钥，从 `.env` 注入 |
| `personal_docs.graph.model` | `none` / `auto` | 图谱抽取模式（none=启发式，auto=LLM） |

## 多智能体流水线

```
用户消息 → Coordinator 路由（判定场景/Direct）
         → (关键词兜底：小模型误判时修正)
         ↓
   非金融 → Direct 直答（快速通道，不回 RAG/审核/风控）
         ↓
   金融   → RAG 检索（公共知识库 + 个人文档）
         → 领域专家作答（单专家 / Multi 圆桌会商）
         → 协调者综合（Multi 模式）
         → 合规审核（建议式，不阻断）
         → 风控检查（建议式，不阻断）
         → 回写修订（吸收审核/风控反馈，输出自然语言）
```

## 关键设计决策

- **统一主库**：07 历史 / 08 个人文档 / 06 知识库全部建表在 `agentar.db`（03 主库），不另开文件
- **conversations.data 单表收口**：聊天正文 + 多专家执行轨迹自包含于一个 JSON 字段，不拆多表
- **G6 → AgentarModel 桥接**：AgentScope 占位模型不实事推理，全部经 `app.model` 统一入口
- **agente inshared 权重**：LLM 与嵌入器共用同一份 `_SHARED_LOADED` 缓存，避免双份加载 OOM
- **旁路历史采集**：07 的 `install_history_tracing()` 无侵入替换 `chat_service.chat`，异步落库失败不阻塞主链路

## 测试

```bash
pytest
```

15 个测试覆盖：模型工厂、嵌入、认证、多智能体框架、RAG、知识库、个人文档、历史记录。
