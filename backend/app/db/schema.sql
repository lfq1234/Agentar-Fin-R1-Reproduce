-- ============================================================
-- Agentar-Fin-R1-Reproduce 后端 · 基础数据库建表脚本（生产用，可执行）
-- 关联：docs/03-传统后端基础层/需求文档.md、技术文档.md
-- 单一事实来源：SQLModel 模型（app/db/models/*.py）；本脚本须与其字段保持一致。
-- 开发/测试期也可用 ORM create_all 代建；生产部署执行本脚本。
-- ============================================================
--
-- 通用约定：
--   - 所有表含 created_at；可更新表额外含 updated_at。
--   - 外键级联删除（ON DELETE CASCADE）。
--   - scene 取值：Banking / Securities / Insurance / Trust / MutualFunds
--   - 时间字段：应用侧写入 UTC（见 models 的 default_factory），此处用 TEXT 存 ISO 字符串。
--   - SQLite 需执行 PRAGMA foreign_keys=ON 才会级联（见 connection.py 已注册事件）。

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

-- 四、agent_traces（多智能体轨迹，预留，03 暂不启用落库）
CREATE TABLE IF NOT EXISTS agent_traces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER,
    trace_json  TEXT,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

-- ============================================================
-- PostgreSQL 适配（生产若是 PG）：
--   1) id 改为：id BIGSERIAL PRIMARY KEY
--   2) 时间字段建议改为 TIMESTAMPTZ NOT NULL（替代 TEXT），应用侧仍写 UTC
--   3) agent_traces.trace_json 建议改为 JSONB（替代 TEXT）
--   4) 外键 / 索引 / 字段名 / 级联语义保持一致即可。
-- 示例（PG 版 users）：
--   CREATE TABLE IF NOT EXISTS users (
--       id          BIGSERIAL PRIMARY KEY,
--       username    TEXT NOT NULL UNIQUE,
--       email       TEXT,
--       created_at  TIMESTAMPTZ NOT NULL,
--       updated_at  TIMESTAMPTZ NOT NULL
--   );
-- ============================================================
