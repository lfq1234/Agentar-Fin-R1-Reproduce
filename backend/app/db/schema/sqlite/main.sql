-- ============================================================
-- Agentar-Fin-R1-Reproduce 后端 · SQLite 建表脚本（生产用，可执行）
-- 关联：docs/03-传统后端基础层、docs/07-会话历史记录-Session
-- 单一事实来源：SQLModel 模型（app/db/models/*.py）+ app/db/history/models.py；
-- 本脚本须与二者字段保持一致。开发/测试期也可用 ORM create_all 代建。
--
-- 说明：本文件合并「03 主库（users / conversations）」与「07 会话历史」，
-- 二者均落同一 SQLite 主库（agentar.db）。按「会话记录极简收口」原则，
-- 聊天记录（用户/助手消息 + 多专家执行轨迹）全部自包含于 conversations.data(JSON) 单字段，
-- 不再拆分 messages / session_meta / session_traces / trace_events / history_embeddings 等表。
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

-- 二、conversations（会话：单表记录全部聊天记录）
--   会话头信息（user_id/scene/title/时间）作为列，聊天正文 + 多专家执行轨迹自包含于 data(JSON)：
--   {"messages":[{role,content,scene,created_at,  -- 用户/助手消息
--                 "trace":{run_id,turn_id,created_at,model,status,duration_ms,
--                          total_tokens,final_result,events:[事件树]}}]}  -- 仅助手消息带 trace
--   一次会话一行（id 主键），不再拆分 messages / session_* / history_embeddings 等表。
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    scene       TEXT,
    title       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    data        TEXT,                        -- JSON：{"messages":[...]}，详见上
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_conversations_user_id ON conversations(user_id);

-- ============================================================
-- 08-个人文档与知识图谱后端（评审 B1：不另开 *.db，四表落同一主库）
-- 关联：docs/08-个人文档与知识图谱后端/技术文档.md §4.1
-- 向量表 personal_doc_chunk_embeddings 定义在 schema/duckdb/personal_docs.sql，
--   同样建在本主库文件内（DuckDB 经 ATTACH 挂载做 list_cosine_distance 检索）。
-- ============================================================

-- 八、personal_documents（个人文档元数据；status: pending|parsing|done|error）
CREATE TABLE IF NOT EXISTS personal_documents (
    id          TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    filename    TEXT NOT NULL,
    size        INTEGER NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    uploaded_at TEXT NOT NULL,
    summary     TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pdoc_user ON personal_documents(user_id);

-- 九、personal_doc_chunks（切分块，复用 06/04 chunking）
CREATE TABLE IF NOT EXISTS personal_doc_chunks (
    id       TEXT PRIMARY KEY,
    doc_id   TEXT NOT NULL,
    user_id  INTEGER NOT NULL,
    idx      INTEGER NOT NULL,
    text     TEXT NOT NULL,
    chapter  TEXT
);
CREATE INDEX IF NOT EXISTS ix_pchunk_doc ON personal_doc_chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_pchunk_user ON personal_doc_chunks(user_id);

-- 十、personal_graph_nodes（图谱节点）
CREATE TABLE IF NOT EXISTS personal_graph_nodes (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    user_id         INTEGER NOT NULL,
    label           TEXT NOT NULL,
    type            TEXT NOT NULL,
    source_doc_id   TEXT NOT NULL,
    properties_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_pgn_doc ON personal_graph_nodes(doc_id);

-- 十一、personal_graph_edges（图谱边）
CREATE TABLE IF NOT EXISTS personal_graph_edges (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    user_id         INTEGER NOT NULL,
    source          TEXT NOT NULL,
    target          TEXT NOT NULL,
    label           TEXT NOT NULL,
    source_doc_id   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pge_doc ON personal_graph_edges(doc_id);
