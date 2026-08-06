"""08 个人文档：向量写入与检索（DuckDB 挂载 03 主库）。

与 06 ``app/db/knowledge/store.py`` 同机制、同库、**同一条连接**：
``app.db.duckdb_conn.get_attached(main_db, alias="kb")``。因此
``kb.chunks``（公共知识库）与 ``kb.personal_doc_chunk_embeddings``（个人文档）
处于同一 catalog，04 的 RAG 可在一次检索内联合召回（评审 B2），且写入天然串行。

DDL 单一事实源：``app/db/schema/duckdb/personal_docs.sql``（运行时读取，不内联）。
SQLite 无数组类型，``embedding`` 以 JSON 文本存储，检索时 ``::FLOAT[]`` 转型。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from app.db import duckdb_conn
from app.db.knowledge.chunking import InconsistentDimensionError, Passage

_SCHEMA_TEXT = (
    Path(__file__).resolve().parents[1] / "schema/duckdb/personal_docs.sql"
).read_text(encoding="utf-8")


def _strip_sql_comments(stmt: str) -> str:
    """剔除整行 ``--`` 注释。注释里出现的 CREATE TABLE 字样不得被当成语句。"""
    return "\n".join(
        line for line in stmt.splitlines() if not line.strip().startswith("--")
    ).strip()


# 先去注释再按 ; 切分（与 app/db/personal_docs/store.py 摘 main.sql 的做法一致）
_SCHEMA_STMTS = [
    body + ";"
    for body in (_strip_sql_comments(s) for s in _SCHEMA_TEXT.split(";"))
    if body.upper().startswith("CREATE TABLE")
]
EMBEDDINGS_DDL = _SCHEMA_STMTS[0]


class PersonalDocVectorIndex:
    """个人文档向量索引（写入 + 用户作用域检索）。"""

    def __init__(self, db_path: str, *, dim: int = 0) -> None:
        self._path = db_path
        self._fixed_dim = dim if dim and dim > 0 else None
        self._conn = duckdb_conn.get_attached(db_path, alias="kb")
        self._init_schema()
        if self._fixed_dim is None:
            row = self._conn.execute(
                "SELECT MAX(dim) FROM kb.personal_doc_chunk_embeddings"
            ).fetchone()
            self._dim = row[0] if row and row[0] is not None else None
        else:
            self._dim = self._fixed_dim

    def _init_schema(self) -> None:
        self._conn.execute(
            EMBEDDINGS_DDL.replace(
                "personal_doc_chunk_embeddings",
                "kb.personal_doc_chunk_embeddings",
                1,
            )
        )
        for sql in (
            "CREATE INDEX IF NOT EXISTS idx_pemb_doc "
            "ON kb.personal_doc_chunk_embeddings (doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_pemb_user "
            "ON kb.personal_doc_chunk_embeddings (user_id)",
        ):
            try:
                self._conn.execute(sql)
            except Exception:
                pass  # 索引失败不阻断建库，降级为列扫描

    # —— 写入 —— #
    def upsert(
        self,
        doc_id: str,
        user_id: int,
        chunk_ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """写入该文档全部块向量（先删旧后插，单文档事务）；返回写入条数。"""
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids 与 vectors 长度不一致")
        for v in vectors:
            if self._dim is None:
                self._dim = len(v)
            if self._dim != len(v):
                raise InconsistentDimensionError(
                    f"embedding 维度不一致：库内 {self._dim}，当前 {len(v)}；"
                    f"切换 embedder 需重建个人文档向量。"
                )
            if self._fixed_dim and self._fixed_dim != len(v):
                raise InconsistentDimensionError(
                    f"embedding 维度与 personal_docs.embed_dim={self._fixed_dim} 不符："
                    f"当前 {len(v)}。"
                )
        try:
            self._conn.execute("BEGIN TRANSACTION")
            self._conn.execute(
                "DELETE FROM kb.personal_doc_chunk_embeddings WHERE doc_id = ?", [doc_id]
            )
            for cid, vec in zip(chunk_ids, vectors):
                self._conn.execute(
                    "INSERT INTO kb.personal_doc_chunk_embeddings "
                    "(chunk_id, doc_id, user_id, embedding, dim) VALUES (?,?,?,?,?)",
                    [cid, doc_id, user_id, json.dumps(list(vec), ensure_ascii=False), len(vec)],
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return len(chunk_ids)

    def delete_doc(self, doc_id: str) -> int:
        n = self._conn.execute(
            "SELECT count(*) FROM kb.personal_doc_chunk_embeddings WHERE doc_id = ?",
            [doc_id],
        ).fetchone()[0]
        self._conn.execute(
            "DELETE FROM kb.personal_doc_chunk_embeddings WHERE doc_id = ?", [doc_id]
        )
        return n

    # —— 检索 —— #
    def vector_search(
        self, query_vec: Sequence[float], user_id: int, top_k: int = 5
    ) -> list[Passage]:
        """用户作用域向量检索（余弦在 DuckDB 端算）；仅返回 status=done 的文档块。"""
        sql = (
            "SELECT c.text, d.filename, c.doc_id, coalesce(c.chapter, '') AS chapter, "
            "coalesce(d.uploaded_at, '') AS uploaded_at, "
            "1 - list_cosine_distance(e.embedding::FLOAT[], ?::FLOAT[]) AS similarity "
            "FROM kb.personal_doc_chunk_embeddings e "
            "JOIN kb.personal_doc_chunks c ON c.id = e.chunk_id "
            "JOIN kb.personal_documents d ON d.id = c.doc_id "
            "WHERE e.user_id = ? AND d.status = 'done' "
            "ORDER BY similarity DESC LIMIT ?"
        )
        rows = self._conn.execute(sql, [list(query_vec), user_id, top_k]).fetchall()
        return [
            Passage(
                content=r[0],
                source=r[1] or r[2],
                score=round(float(r[5]), 4),
                chapter=r[3],
                effective_date=r[4],
                title=r[1] or "",
                doc_type="personal",
                doc_id=r[2] or "",
            )
            for r in rows
        ]

    def stats(self, user_id: Optional[int] = None) -> dict:
        if user_id is None:
            row = self._conn.execute(
                "SELECT count(*), MAX(dim) FROM kb.personal_doc_chunk_embeddings"
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT count(*), MAX(dim) FROM kb.personal_doc_chunk_embeddings "
                "WHERE user_id = ?",
                [user_id],
            ).fetchone()
        return {"vector_count": row[0] or 0, "dim": row[1]}
