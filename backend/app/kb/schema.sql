-- 06 知识库 SQLite schema（单一真相来源，store.py / sqlite_store.py 内联复用同一 DDL）
--
-- 架构约定：文件（文档 / 块 / 向量）全部落盘到 SQLite；DuckDB 引擎通过 ATTACH 挂载
-- 本文件做向量检索（list_cosine_distance(embedding::FLOAT[], ?)），写入亦走本表。
-- embedding 以 JSON 数组文本存储，检索时由 DuckDB 在 SQL 端转型为 FLOAT[]。

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY,
    doc_id          TEXT,
    title           TEXT,
    doc_type        TEXT,
    agency          TEXT,
    domain          TEXT,            -- 逗号分隔多标签（检索时用 string_split + list_contains 精确命中）
    chapter         TEXT,
    content         TEXT,
    effective_date  TEXT,
    version         TEXT,
    source_path     TEXT,
    embedding       TEXT,            -- JSON 数组（FLOAT 列表）
    dim             INTEGER
);

CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_eff    ON chunks (effective_date);

-- 构建/重扫时间记录（stats() 读取）
CREATE TABLE IF NOT EXISTS kb_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
