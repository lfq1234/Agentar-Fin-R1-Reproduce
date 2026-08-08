# frontend

React 18 + Vite 5 + TypeScript 交互前端，对接后端多智能体金融对话系统。

## 功能

- **SSE 流式多智能体对话**：每个领域专家依次出现（🏦银行/📈证券/🛡️保险/🏛️信托/🧺基金），@ 下一个专家，合规审核 & 风控逐条建议，协调者最终修订
- **个人文档面板**（双标签）：
  - 文档：上传（txt/md/pdf/docx）→ 自动解析 → 切分 → 向量化 → 知识图谱抽取
  - 知识图谱：@antv/g6 力导向渲染，按类型/文档筛选，双击查看实体详情
- **会话历史侧边栏**：登录后从后端拉取历史，支持删除、续聊
- **用户系统**：注册/登录（JWT Bearer），数据隔离

## 运行

```bash
npm install
npm run dev        # http://localhost:5173
```

`vite.config.ts` 已配置 `/api` 代理到 `http://localhost:8000`。

## 构建

```bash
npm run build      # 产物在 dist/
npm run preview
```

## 目录结构

```
src/
├── main.tsx                  # React 入口
├── App.tsx                   # 根组件：登录门禁 + 三态布局
├── index.css                 # 全局样式
├── api/
│   └── client.ts             # 统一 API 客户端（fetch + JWT + SSE 流式）
├── types/
│   └── agent.ts              # 全部 TS 类型 + AGENT_DISPLAY 映射表
├── hooks/
│   ├── useChat.ts            # SSE 流式聊天 + 会话管理
│   ├── useKnowledgeGraph.ts  # 知识图谱数据 + G6 缩放控制
│   └── usePersonalDocs.ts    # 文档上传/列表操作
├── components/
│   ├── ChatInput.tsx         # 输入框
│   ├── MessageList.tsx       # 消息列表
│   ├── MessageBubble.tsx     # 用户/智能体/助手气泡
│   ├── Sidebar.tsx           # 侧边栏容器
│   ├── SessionList.tsx       # 会话列表
│   ├── PersonalDocsPanel.tsx # 个人文档双标签面板
│   ├── DocumentUploader.tsx  # 文档上传器
│   ├── DocumentList.tsx      # 文档列表
│   ├── KnowledgeGraphViewer.tsx  # G6 知识图谱画布
│   ├── GraphControls.tsx     # 缩放+类型/文档筛选
│   └── ...
├── mock/
│   └── personalDocs.ts       # Mock 数据（VITE_USE_MOCK=true）
└── utils/
    └── graph.ts              # 图谱样式工具
```

## 接口契约

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| POST | `/api/v1/chat` | 同步多智能体对话 |
| POST | `/api/v1/chat/stream` | **SSE 流式**多智能体对话 |
| POST | `/api/v1/analyze` | 结构化分析 |
| POST | `/api/v1/auth/register` | 注册 |
| POST | `/api/v1/auth/login/access-token` | OAuth2 登录 |
| GET | `/api/v1/auth/me` | 当前用户 |
| GET | `/api/v1/history/sessions` | 会话列表 |
| GET | `/api/v1/history/sessions/:id` | 会话详情 |
| DELETE | `/api/v1/history/sessions/:id` | 删除会话 |
| POST | `/api/v1/documents` | 上传文档（multipart） |
| GET | `/api/v1/documents` | 文档列表 |
| DELETE | `/api/v1/documents/:id` | 删除文档 |
| GET | `/api/v1/knowledge-graph` | 知识图谱 |
| POST | `/api/v1/rag/retrieve` | RAG 检索自检 |
