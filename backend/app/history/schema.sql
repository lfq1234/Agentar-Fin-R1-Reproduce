-- ============================================================
-- 07-会话历史记录（Session History & Trace）建表脚本（复用 03 主库 agentar.db）
-- 关联：docs/07-会话历史记录-Session/需求文档.md、技术文档.md、评审文档.md
-- 单一事实来源：app/history/models.py（dataclass）的字段须与本脚本保持一致。
--
-- 设计约定：
--   - 时间字段统一 INTEGER（epoch 毫秒，UTC），避免 VARCHAR ISO 比较脆弱（评审 B4）。
--   - conversation_id / user_id 以 TEXT 存储（03 侧为 INTEGER，07 侧统一转 str 以兼容
--     同库引用与无侵入接入；查询 03 时在 store 内按需 int() 转换）。
--   - trace_events 为 append-only：应用层禁 UPDATE / 单条 DELETE，仅整清理。
--   - 07 表与 03 同库（agentar.db），service/view 分层；03 预留 agent_traces 命名废弃，
--     统一采用 07 的 session_traces + trace_events（评审 S6）。
-- ============================================================

-- 一、session_meta（会话扩展元信息，07 侧表，不 ALTER 03 conversations）
CREATE TABLE IF NOT EXISTS session_meta (
    conversation_id TEXT PRIMARY KEY,   -- 关联 03 conversations.id
    scene           TEXT,
    title           TEXT,
    status          TEXT DEFAULT 'active',  -- active | archived | deleted
    msg_count       INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    first_at        INTEGER,                -- epoch 毫秒
    last_at         INTEGER,                -- epoch 毫秒
    created_at      INTEGER                 -- epoch 毫秒
);
CREATE INDEX IF NOT EXISTS idx_meta_status ON session_meta(status);

-- 二、session_traces（一次 run() 的轨迹头，落地 03 预留 agent_traces 的语义）
CREATE TABLE IF NOT EXISTS session_traces (
    run_id          TEXT PRIMARY KEY,    -- 07 生成 UUID
    turn_id         TEXT,                -- 一次 user->assistant 交互的稳定键（评审 B5）
    conversation_id TEXT,
    user_id         TEXT,
    scene           TEXT,
    created_at      INTEGER,             -- epoch 毫秒
    duration_ms     INTEGER,
    total_tokens    INTEGER,
    model           TEXT,
    status          TEXT,                -- ok | error
    final_result    TEXT                 -- JSON 序列化 AgentResult
);
CREATE INDEX IF NOT EXISTS idx_traces_conv ON session_traces(conversation_id);
CREATE INDEX IF NOT EXISTS idx_traces_user ON session_traces(user_id);
CREATE INDEX IF NOT EXISTS idx_traces_time ON session_traces(created_at);
CREATE INDEX IF NOT EXISTS idx_traces_turn ON session_traces(turn_id);

-- 三、trace_events（轨迹事件树：agent 步骤 / tool / RAG 命中 / 审核 / 风控 / 错误 / 主消息）
CREATE TABLE IF NOT EXISTS trace_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    parent_event_id INTEGER,             -- 层级（NULL=根）
    turn_id         TEXT,                -- 关联一次 user->assistant 交互
    seq             INTEGER DEFAULT 0,   -- run 内单调序号，保证异步环境下的顺序（评审 S3）
    agent           TEXT,
    type            TEXT,                -- user|assistant|tool_call|tool_result|rag_hit|review|risk|error
    summary_in      TEXT,
    summary_out     TEXT,
    meta_json       TEXT,                -- tool 名/参数、RAG 命中溯源(Passage 字段)、错误类型等
    tokens          INTEGER DEFAULT 0,
    duration_ms     INTEGER DEFAULT 0,
    created_at      INTEGER              -- epoch 毫秒
);
CREATE INDEX IF NOT EXISTS idx_events_run ON trace_events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_turn ON trace_events(turn_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON trace_events(type);

-- 四、history_embeddings（可选语义检索：历史消息向量，开启 history.semantic_search 启用）
CREATE TABLE IF NOT EXISTS history_embeddings (
    message_id      TEXT PRIMARY KEY,    -- 关联 03 messages.id
    conversation_id TEXT,
    embedding       TEXT,                -- JSON 数组
    dim             INTEGER,
    model           TEXT                 -- embedder 版本（评审 S5：切换需重建）
);
