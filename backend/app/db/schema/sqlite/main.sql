-- ============================================================
-- Agentar-Fin-R1-Reproduce 后端 · SQLite 建表脚本（生产用，可执行）
-- 关联：docs/03-传统后端基础层、docs/07-会话历史记录-Session
-- 单一事实来源：SQLModel 模型（app/db/models/*.py）+ app/db/history/models.py；
-- 本脚本须与二者字段保持一致。开发/测试期也可用 ORM create_all 代建。
--
-- 说明：本文件合并「03 主库（users / conversations / messages）」与
-- 「07 会话历史（session_meta / session_traces / trace_events / history_embeddings）」，
-- 二者均落同一 SQLite 主库（agentar.db）。03 预留的 agent_traces 表已废弃，
-- 统一由 07 的 session_traces + trace_events 接管（评审 S6）。
-- ============================================================
--
-- 通用约定：
--   - 03 侧时间字段用 TEXT 存 ISO（UTC）；07 侧用 INTEGER 存 epoch 毫秒（评审 B4）。
--   - 外键级联删除（ON DELETE CASCADE），SQLite 需 PRAGMA foreign_keys=ON（见 connection.py）。

-- 一、users（用户，最小字段，03 不接鉴权）
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT NOT NULL UNIQUE,
    email       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- 二、conversations（会话）
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    scene       TEXT,
    title       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_conversations_user_id ON conversations(user_id);

-- 三、messages（消息）
CREATE TABLE IF NOT EXISTS messages (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id   INTEGER NOT NULL,
    role              TEXT NOT NULL,
    content           TEXT NOT NULL,
    scene             TEXT,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_messages_conversation_id ON messages(conversation_id);

-- 四、session_meta（07 会话扩展元信息，不 ALTER 03 conversations）
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

-- 五、session_traces（一次 run() 的轨迹头，接管 03 预留 agent_traces 语义）
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

-- 六、trace_events（轨迹事件树：agent 步骤 / tool / RAG 命中 / 审核 / 风控 / 错误 / 主消息）
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

-- 七、history_embeddings（可选语义检索：历史消息向量，开启 history.semantic_search 启用）
CREATE TABLE IF NOT EXISTS history_embeddings (
    message_id      TEXT PRIMARY KEY,    -- 关联 03 messages.id
    conversation_id TEXT,
    embedding       TEXT,                -- JSON 数组
    dim             INTEGER,
    model           TEXT                 -- embedder 版本（评审 S5：切换需重建）
);
