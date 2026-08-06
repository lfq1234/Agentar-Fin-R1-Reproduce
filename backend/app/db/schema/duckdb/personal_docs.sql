-- ============================================================
-- 08 个人文档 embeddings 建表脚本（DuckDB 引擎作用于 SQLite 主库 agentar.db）
-- 关联：docs/08-个人文档与知识图谱后端/技术文档.md §4.2、评审文档 B1
--
-- 唯一事实来源：本文件；app/db/personal_docs/vector.py 运行时读取并按
--   CREATE TABLE 切分执行，不再内联 DDL，避免 schema 漂移（同 06 knowledge.sql）。
--
-- 架构约定同 06：表落 SQLite 主库，DuckDB 经 ATTACH 挂载做
--   list_cosine_distance(embedding::FLOAT[], ?) 余弦近邻；
--   SQLite 无数组类型，embedding 以 JSON 数组文本（TEXT）存储，检索时 SQL 端转型。
--
-- 与 06 chunks 表同库同连接（alias kb），故 04 的 RAG 可在一次查询内
--   同时覆盖「公共知识库 + 个人文档」两个来源（评审 B2 用户作用域检索）。
-- ============================================================

CREATE TABLE IF NOT EXISTS personal_doc_chunk_embeddings (
    chunk_id  TEXT PRIMARY KEY,        -- 关联 personal_doc_chunks.id
    doc_id    TEXT NOT NULL,
    user_id   INTEGER NOT NULL,
    embedding TEXT NOT NULL,           -- JSON 数组（FLOAT 列表）
    dim       INTEGER
);
