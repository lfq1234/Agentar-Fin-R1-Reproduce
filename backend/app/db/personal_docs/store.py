"""08 个人文档：SQLite 主库真相源（结构化数据读写）。

层位：与 06 ``app/db/knowledge/sqlite_store.py`` 同层——db 包下的存储实现，
只做数据访问，不含编排（ingest 流水线在 ``app/services/documents_service.py``）。

关键约定：
- **不另开 \\*.db**：直接连 ``config.db.sqlite_path``（默认 ``./agentar.db``），
  与 03 users/conversations/messages、07 会话历史同库（评审 B1）。
- **DDL 单一事实源**：建表语句从 ``app/db/schema/sqlite/main.sql`` 读取并筛出
  ``personal_*`` 相关语句执行，不在代码里内联（同 06 读 ``knowledge.sql`` 的范式）。
- **应用层级联**：``delete_document`` 显式清 文档 / 块 / 向量 / 图谱节点 / 图谱边 五处，
  覆盖图谱表未设外键的缺口（技术文档 §13）。
- **作用域强制**：所有读写都带 ``user_id`` 条件，跨用户不可见。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

# —— DDL 单一事实源：从 schema/sqlite/main.sql 摘取 personal_* 语句 —— #
_SCHEMA_TEXT = (
    Path(__file__).resolve().parents[1] / "schema/sqlite/main.sql"
).read_text(encoding="utf-8")


def _strip_sql_comments(stmt: str) -> str:
    return "\n".join(
        line for line in stmt.splitlines() if not line.strip().startswith("--")
    ).strip()


_PERSONAL_STMTS: list[str] = [
    body + ";"
    for body in (_strip_sql_comments(s) for s in _SCHEMA_TEXT.split(";"))
    if body.upper().startswith(("CREATE TABLE", "CREATE INDEX")) and "personal_" in body
]

# 表清单（删除时应用层级联的目标；向量表由 vector.py 负责）
_TABLES = (
    "personal_graph_edges",
    "personal_graph_nodes",
    "personal_doc_chunks",
    "personal_documents",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_doc_id() -> str:
    return uuid.uuid4().hex


class PersonalDocStore:
    """个人文档 / 块 / 图谱在 03 主库中的读写入口。"""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        # FastAPI 同一事件循环下可能跨线程调用（run_in_threadpool），故关闭线程亲和检查，
        # 由 _lock 串行化写入；WAL 保证与 03/07 的异步引擎共存不互锁。
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            for stmt in _PERSONAL_STMTS:
                self._conn.execute(stmt)
            self._conn.commit()

    # —— 文档元数据 —— #
    def create_doc(self, doc_id: str, user_id: int, filename: str, size: int) -> None:
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT INTO personal_documents "
                "(id, user_id, filename, size, status, error, uploaded_at, summary, "
                " created_at, updated_at) "
                "VALUES (?,?,?,?,'pending',NULL,?,NULL,?,?)",
                [doc_id, user_id, filename, size, now, now, now],
            )
            self._conn.commit()

    def set_status(
        self,
        doc_id: str,
        status: str,
        *,
        error: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE personal_documents SET status = ?, "
                "error = COALESCE(?, error), summary = COALESCE(?, summary), "
                "updated_at = ? WHERE id = ?",
                [status, error, summary, _now_iso(), doc_id],
            )
            self._conn.commit()

    def get_doc(self, doc_id: str, user_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id, filename, size, status, error, uploaded_at, summary "
            "FROM personal_documents WHERE id = ? AND user_id = ?",
            [doc_id, user_id],
        ).fetchone()
        return dict(row) if row else None

    def list_docs(self, user_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, filename, size, status, error, uploaded_at, summary "
            "FROM personal_documents WHERE user_id = ? "
            "ORDER BY uploaded_at DESC, rowid DESC",
            [user_id],
        ).fetchall()
        return [dict(r) for r in rows]

    # —— 块 —— #
    def add_chunks(
        self, doc_id: str, user_id: int, chunks: Iterable
    ) -> list[str]:
        """写入切分块，返回与入参同序的 chunk_id 列表（供向量表关联）。"""
        ids: list[str] = []
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                self._conn.execute(
                    "DELETE FROM personal_doc_chunks WHERE doc_id = ?", [doc_id]
                )
                for i, c in enumerate(chunks):
                    cid = f"{doc_id}:c{i}"
                    ids.append(cid)
                    self._conn.execute(
                        "INSERT INTO personal_doc_chunks "
                        "(id, doc_id, user_id, idx, text, chapter) VALUES (?,?,?,?,?,?)",
                        [cid, doc_id, user_id, i, c.text, getattr(c, "chapter", "")],
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return ids

    def count_chunks(self, doc_id: str) -> int:
        return self._conn.execute(
            "SELECT count(*) FROM personal_doc_chunks WHERE doc_id = ?", [doc_id]
        ).fetchone()[0]

    # —— 图谱 —— #
    def add_graph(self, doc_id: str, user_id: int, nodes: Iterable, edges: Iterable) -> None:
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                self._conn.execute(
                    "DELETE FROM personal_graph_edges WHERE doc_id = ?", [doc_id]
                )
                self._conn.execute(
                    "DELETE FROM personal_graph_nodes WHERE doc_id = ?", [doc_id]
                )
                for n in nodes:
                    self._conn.execute(
                        "INSERT INTO personal_graph_nodes "
                        "(id, doc_id, user_id, label, type, source_doc_id, properties_json) "
                        "VALUES (?,?,?,?,?,?,?)",
                        [
                            n.id,
                            doc_id,
                            user_id,
                            n.label,
                            n.type,
                            n.source_doc_id,
                            json.dumps(n.properties or {}, ensure_ascii=False),
                        ],
                    )
                for e in edges:
                    self._conn.execute(
                        "INSERT INTO personal_graph_edges "
                        "(id, doc_id, user_id, source, target, label, source_doc_id) "
                        "VALUES (?,?,?,?,?,?,?)",
                        [e.id, doc_id, user_id, e.source, e.target, e.label, e.source_doc_id],
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def get_graph(self, user_id: int) -> tuple[list[dict], list[dict]]:
        """返回该用户全部已完成文档的图谱（nodes, edges）。"""
        nodes = [
            {
                "id": r["id"],
                "label": r["label"],
                "type": r["type"],
                "source_doc_id": r["source_doc_id"],
                "properties": json.loads(r["properties_json"] or "{}"),
            }
            for r in self._conn.execute(
                "SELECT n.id, n.label, n.type, n.source_doc_id, n.properties_json "
                "FROM personal_graph_nodes n "
                "JOIN personal_documents d ON d.id = n.doc_id "
                "WHERE n.user_id = ? AND d.status = 'done' ORDER BY n.rowid",
                [user_id],
            ).fetchall()
        ]
        edges = [
            {
                "id": r["id"],
                "source": r["source"],
                "target": r["target"],
                "label": r["label"],
                "source_doc_id": r["source_doc_id"],
            }
            for r in self._conn.execute(
                "SELECT e.id, e.source, e.target, e.label, e.source_doc_id "
                "FROM personal_graph_edges e "
                "JOIN personal_documents d ON d.id = e.doc_id "
                "WHERE e.user_id = ? AND d.status = 'done' ORDER BY e.rowid",
                [user_id],
            ).fetchall()
        ]
        return nodes, edges

    # —— 生命周期 —— #
    def delete_document(self, doc_id: str, user_id: int) -> bool:
        """应用层级联删除四张结构化表；向量表由调用方（service）经 vector.delete_doc 清理。

        Returns:
            `bool`: 该 user 下确有此文档并删除成功时 True；不存在 / 越权时 False。
        """
        if self.get_doc(doc_id, user_id) is None:
            return False
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                for table in _TABLES:
                    # personal_documents 的主键列名是 id，其余从表用 doc_id 关联
                    key_col = "id" if table == "personal_documents" else "doc_id"
                    self._conn.execute(
                        f"DELETE FROM {table} WHERE {key_col} = ? AND user_id = ?",
                        [doc_id, user_id],
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return True

    def stats(self, user_id: int) -> dict:
        row = self._conn.execute(
            "SELECT count(*), coalesce(sum(size), 0) FROM personal_documents WHERE user_id = ?",
            [user_id],
        ).fetchone()
        chunks = self._conn.execute(
            "SELECT count(*) FROM personal_doc_chunks WHERE user_id = ?", [user_id]
        ).fetchone()[0]
        return {"doc_count": row[0], "total_size": row[1], "chunk_count": chunks}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
