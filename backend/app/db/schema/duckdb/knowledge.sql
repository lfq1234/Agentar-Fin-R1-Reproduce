-- ============================================================
-- 06 知识库 chunks / kb_meta 建表脚本（DuckDB 引擎作用于 SQLite 文件）
-- 关联：docs/06-知识库-DuckDB/需求文档.md、技术文档.md
-- 唯一事实来源：本文件；app/db/knowledge/sqlite_store.py 与 store.py 均读取本文件，
--   不再内联 DDL，避免 schema 漂移。
--
-- 架构约定：文档 / 块 / 向量全部落盘到 SQLite；DuckDB 引擎通过 ATTACH 挂载本文件做
--   向量检索（list_cosine_distance(embedding::FLOAT[], ?)），写入亦走本表。
-- embedding 以 JSON 数组文本存储，检索时由 DuckDB 在 SQL 端转型为 FLOAT[]。
--
-- 索引：本文件仅含表定义。sqlite_store（kb.engine=sqlite）不建索引；DuckDB store
--   （kb.engine=duckdb）按引擎自建国前缀索引（idx_doc_id / idx_eff ON kb.chunks）。
-- ============================================================

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

CREATE TABLE IF NOT EXISTS kb_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
