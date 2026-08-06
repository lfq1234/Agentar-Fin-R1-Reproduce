"""06-知识库（DuckDB）向量检索后端核心。

架构（按约定「文件放 SQLite、向量检索放 DuckDB」）：
- **文件存储**：所有文档/块/embedding 落盘到 **SQLite** 文件（``kb.path`` 指向的 .db）；
- **向量检索**：DuckDB 以 ``ATTACH`` 挂载该 SQLite 文件（READ_WRITE），在 SQL 端用
  ``list_cosine_distance(embedding::FLOAT[], ?)`` 做余弦近邻，元数据预过滤先缩小候选集；
- 双路独立召回（向量 + BM25）后取**并集**融合重排（评审 B1 修复）。

切分 / BM25 / 归一化 / Passage 复用 ``app.db.knowledge.chunking``（纯 Python，无 torch 依赖）；
SQLite schema（``CHUNKS_DDL``）由 ``app.db.knowledge.sqlite_store`` 单一定义，两后端共用。

设计文档关联：06 技术文档 §3（schema/向量检索）、§5（召回与重排）、§6（生命周期）。
"""
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

from app.db import duckdb_conn
from app.db.knowledge.chunking import (
    Passage,
    InconsistentDimensionError,
    _BM25,
    _normalize,
    _split_into_chunks,
)
from app.db.knowledge.sqlite_store import CHUNKS_DDL, KB_META_DDL


def _domain_tags(domain: str) -> list[str]:
    """把逗号分隔的 domain 多标签规范化为数组（过滤用，精确命中）。"""
    return [t.strip() for t in domain.split(",") if t.strip()]


@runtime_checkable
class KnowledgeStore(Protocol):
    """06 知识库双后端统一契约（需求文档 §6.1）。"""

    def ingest_text(
        self,
        text: str,
        *,
        doc_id: str,
        title: str = "",
        doc_type: str = "internal",
        agency: str = "",
        domain: str = "",
        effective_date: str = "",
        version: str = "1",
        source_path: str = "",
        max_chars: int = 600,
        overlap: int = 80,
    ) -> int: ...

    def ingest_document(self, path: str, **meta) -> int: ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        effective_after: Optional[str] = None,
        vec_weight: float = 0.7,
    ) -> list[Passage]: ...

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        *,
        user_id: Optional[int] = None,
        use_personal_docs: bool = False,
    ) -> list[Passage]: ...

    def vector_search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        domain: Optional[str] = None,
        effective_after: Optional[str] = None,
    ) -> list[Passage]: ...

    def stats(self) -> dict: ...

    def list_docs(self) -> list[dict]: ...

    def delete_doc(self, doc_id: str) -> int: ...

    def rebuild(self, *, rescan: bool = False) -> int: ...


@dataclass
class _Row:
    """召回候选行（轻量载体，避免依赖 DuckDB 行顺序）。"""

    id: int
    doc_id: str
    title: str
    agency: str
    doc_type: str
    domain: str
    version: str
    chapter: str
    content: str
    effective_date: str


class DuckDBKnowledgeStore:
    """DuckDB 知识库存储后端（06 默认实现）。

    关键特性：
    - 向量余弦在 SQL 端算（``list_cosine_distance``），元数据预过滤先缩小候选集；
    - 双路独立召回（向量 + BM25）后取**并集**融合重排（评审 B1 修复：BM25 不在
      向量候选子集上算，真正补回向量漏召的块）；
    - 维度自适应（首个 ingest 定维，不一致显式报错）；
    - 可选 vss HNSW 加速（默认关闭，离线降级精确扫描）；
    - embedder 可注入（无 torch 单测用 mock），未注入时懒加载 01 ``get_embedder``。
    """

    def __init__(
        self,
        sqlite_path: str,
        *,
        embedder=None,
        dim: int = 0,
        use_vss: bool = False,
        chunk_max_chars: int = 600,
        chunk_overlap: int = 80,
        kb_dir: Optional[str] = None,
        reranker=None,
    ) -> None:
        self._path = sqlite_path
        self._embedder = embedder  # 可注入（测试用）；None 时懒加载 01
        # 维度：config 显式固定（>0）或自适应（None，首个 ingest 决定）
        self._fixed_dim = dim if dim and dim > 0 else None
        self._use_vss = bool(use_vss)  # 注：挂载的 SQLite 表无法建原生 HNSW，不生效
        self._chunk_max_chars = chunk_max_chars
        self._chunk_overlap = chunk_overlap
        self._kb_dir = kb_dir
        self._reranker = reranker
        # 文件存储：SQLite 文件（若不存在先建空库），再由 DuckDB ATTACH 挂载做向量检索。
        # 评审 B1：默认指向 03 主库 agentar.db，与 08 个人文档共用同一条 DuckDB 连接
        # （见 app/db/duckdb_conn.py），避免双写者竞争，也让两类语料同 catalog 可联合检索。
        self._conn = duckdb_conn.get_attached(sqlite_path, alias="kb")
        self._init_schema()
        # 维度自适应：从库内已有数据 seed（跨进程一致）
        if self._fixed_dim is None:
            row = self._conn.execute("SELECT MAX(dim) FROM kb.chunks").fetchone()
            self._dim = row[0] if row and row[0] is not None else None
        else:
            self._dim = self._fixed_dim

    # —— schema —— #
    def _init_schema(self) -> None:
        # 挂载的 SQLite 文件内建表（单一来源 CHUNKS_DDL，前缀 kb.）
        self._conn.execute(CHUNKS_DDL.replace("chunks", "kb.chunks", 1))
        self._conn.execute(KB_META_DDL.replace("kb_meta", "kb.kb_meta", 1))
        # 索引创建失败不应阻断建库，降级为列扫描
        for sql in (
            "CREATE INDEX IF NOT EXISTS idx_doc_id ON kb.chunks (doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_eff ON kb.chunks (effective_date)",
        ):
            try:
                self._conn.execute(sql)
            except Exception:
                pass
        # 注：挂载的 SQLite 表无法建 DuckDB 原生 HNSW 索引，use_vss 在此不生效；
        # 向量检索走 list_cosine_distance 精确扫描（万级规模足够）。

    # —— embedder —— #
    def set_embedder(self, fn) -> None:
        """注入 embedding 函数（测试或自定义实现用）。"""
        self._embedder = fn

    def _get_embedder(self):
        if self._embedder is None:
            # 函数内惰性导入，避免 import 时拉起 app.model（local 实现依赖 torch）。
            from app.model import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._get_embedder().embed(texts)

    # —— ingest —— #
    def ingest_text(
        self,
        text: str,
        *,
        doc_id: str,
        title: str = "",
        doc_type: str = "internal",
        agency: str = "",
        domain: str = "",
        effective_date: str = "",
        version: str = "1",
        source_path: str = "",
        max_chars: int = 600,
        overlap: int = 80,
    ) -> int:
        """切分 + 向量化 + 写入挂载的 SQLite（单 doc 事务包裹，评审 I3）；返回写入块数。"""
        chunks = _split_into_chunks(text, max_chars=max_chars, overlap=overlap)
        if not chunks:
            return 0
        vecs = self._embed([c.text for c in chunks])

        # 维度自适应校验（§3.4）
        for v in vecs:
            if self._dim is None:
                self._dim = len(v)
            if self._dim != len(v):
                raise InconsistentDimensionError(
                    f"embedding 维度不一致：库内 {self._dim}，当前 {len(v)}；"
                    f"切换 embedder 需 rebuild 知识库。"
                )
            if self._fixed_dim and self._fixed_dim != len(v):
                raise InconsistentDimensionError(
                    f"embedding 维度与 kb.embed_dim={self._fixed_dim} 不符：当前 {len(v)}。"
                )

        dom = ",".join(_domain_tags(domain))
        try:
            self._conn.execute("BEGIN TRANSACTION")
            # 增量去重：同 doc_id 视为新版本，先删旧再写（FR2）
            self._conn.execute("DELETE FROM kb.chunks WHERE doc_id = ?", [doc_id])
            # 显式分配 id（DuckDB 经 ATTACH 写入 SQLite 不会自动填充 INTEGER PRIMARY KEY 行别名）
            max_id = self._conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM kb.chunks"
            ).fetchone()[0]
            for i, (c, v) in enumerate(zip(chunks, vecs)):
                self._conn.execute(
                    "INSERT INTO kb.chunks "
                    "(id, doc_id, title, doc_type, agency, domain, chapter, "
                    "effective_date, version, source_path, content, embedding, dim) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        max_id + 1 + i,
                        doc_id,
                        title,
                        doc_type,
                        agency,
                        dom,
                        c.chapter,
                        effective_date,
                        version,
                        source_path,
                        c.text,
                        json.dumps(v, ensure_ascii=False),
                        len(v),
                    ],
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return len(chunks)

    def ingest_document(self, path: str, **meta) -> int:
        """按扩展名接入文档；txt/md/html 原生支持，二进制抛 NotImplementedError。

        md 文件支持 YAML frontmatter（``domain`` / ``agency`` / ``doc_type`` /
        ``effective_date`` / ``version`` / ``title``）作为元数据。
        """
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".txt", ".md", ".markdown", ".html", ".htm"):
            raise NotImplementedError(
                f"暂不支持 {ext} 解析；PDF/DOCX/XLSX 为后续需求。"
            )
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        fm: dict = {}
        body = raw
        if ext in (".md", ".markdown"):
            fm, body = _parse_frontmatter(raw)

        if ext in (".html", ".htm"):
            body = re.sub(r"<script[\s\S]*?</script>", " ", body)
            body = re.sub(r"<style[\s\S]*?</style>", " ", body)
            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"&nbsp;", " ", body)
            body = re.sub(r"\s+", " ", body)

        doc_id = meta.pop("doc_id", fm.get("doc_id") or os.path.basename(path))
        title = meta.pop("title", fm.get("title") or os.path.basename(path))
        merged = {
            "doc_type": fm.get("doc_type", "internal"),
            "agency": fm.get("agency", ""),
            "domain": fm.get("domain", ""),
            "effective_date": fm.get("effective_date", ""),
            "version": fm.get("version", "1"),
            "source_path": path,
        }
        merged.update(meta)
        return self.ingest_text(body, doc_id=doc_id, title=title, **merged)

    # —— 召回辅助（统一返回 (row, vec_sim|None, bm25|None)） —— #
    def _duckdb_recall(
        self,
        qvec: list[float],
        k: int,
        domain: Optional[str],
        effective_after: Optional[str],
    ) -> list[tuple[_Row, Optional[float], None]]:
        sql = (
            "SELECT rowid AS id, doc_id, title, agency, doc_type, domain, version, "
            "chapter, content, effective_date, "
            "1 - list_cosine_distance(embedding::FLOAT[], ?::FLOAT[]) AS similarity "
            "FROM kb.chunks "
            "WHERE (? IS NULL OR list_contains(string_split(coalesce(domain, ''), ','), ?)) "
            "  AND (? IS NULL OR effective_date >= ?) "
            "ORDER BY similarity DESC LIMIT ?"
        )
        rows = self._conn.execute(
            sql, [qvec, domain, domain, effective_after, effective_after, k]
        ).fetchall()
        out = []
        for r in rows:
            row = _Row(
                id=r[0], doc_id=r[1], title=r[2] or "", agency=r[3] or "",
                doc_type=r[4] or "", domain=r[5] or "", version=r[6] or "",
                chapter=r[7] or "", content=r[8], effective_date=r[9] or "",
            )
            out.append((row, float(r[10]), None))
        return out

    def _fetch_filtered(
        self, domain: Optional[str], effective_after: Optional[str]
    ) -> list[_Row]:
        sql = (
            "SELECT rowid AS id, doc_id, title, agency, doc_type, domain, version, "
            "chapter, content, effective_date "
            "FROM kb.chunks "
            "WHERE (? IS NULL OR list_contains(string_split(coalesce(domain, ''), ','), ?)) "
            "  AND (? IS NULL OR effective_date >= ?)"
        )
        rows = self._conn.execute(
            sql, [domain, domain, effective_after, effective_after]
        ).fetchall()
        return [
            _Row(
                id=r[0], doc_id=r[1], title=r[2] or "", agency=r[3] or "",
                doc_type=r[4] or "", domain=r[5] or "", version=r[6] or "",
                chapter=r[7] or "", content=r[8], effective_date=r[9] or "",
            )
            for r in rows
        ]

    def _bm25_recall(
        self,
        query: str,
        k: int,
        domain: Optional[str],
        effective_after: Optional[str],
    ) -> list[tuple[_Row, None, float]]:
        rows = self._fetch_filtered(domain, effective_after)
        if not rows:
            return []
        bm25 = _BM25([r.content for r in rows])
        scores = bm25.scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(rows[i], None, float(scores[i])) for i in order]

    # —— 个人文档召回（08，评审 B2：用户作用域） —— #
    # 08 的 personal_doc_chunk_embeddings / personal_doc_chunks 与 kb.chunks 同库同连接，
    # 故可在同一 DuckDB catalog 内检索；synthetic id 取 -rowid，与 kb.chunks 的正 rowid
    # 天然不冲突，可直接进 _union_recall 融合。库内无 08 表时静默返回空（06 独立可用）。
    _PERSONAL_COLS = (
        "-c.rowid AS id, c.doc_id, d.filename AS title, c.text AS content, "
        "coalesce(c.chapter, '') AS chapter, coalesce(d.uploaded_at, '') AS uploaded_at"
    )

    @staticmethod
    def _personal_row(r: Sequence) -> _Row:
        return _Row(
            id=int(r[0]),
            doc_id=r[1] or "",
            title=r[2] or "",
            agency="",
            doc_type="personal",  # 下游据此区分「个人文档 / 公共知识库」来源
            domain="",
            version="",
            chapter=r[4] or "",
            content=r[3],
            effective_date=r[5] or "",
        )

    def _personal_vector_recall(
        self, qvec: list[float], k: int, user_id: int
    ) -> list[tuple[_Row, Optional[float], None]]:
        sql = (
            f"SELECT {self._PERSONAL_COLS}, "
            "1 - list_cosine_distance(e.embedding::FLOAT[], ?::FLOAT[]) AS similarity "
            "FROM kb.personal_doc_chunk_embeddings e "
            "JOIN kb.personal_doc_chunks c ON c.id = e.chunk_id "
            "JOIN kb.personal_documents d ON d.id = c.doc_id "
            "WHERE e.user_id = ? AND d.status = 'done' "
            "ORDER BY similarity DESC LIMIT ?"
        )
        try:
            rows = self._conn.execute(sql, [qvec, user_id, k]).fetchall()
        except Exception:
            return []
        return [(self._personal_row(r), float(r[6]), None) for r in rows]

    def _personal_bm25_recall(
        self, query: str, k: int, user_id: int
    ) -> list[tuple[_Row, None, float]]:
        sql = (
            f"SELECT {self._PERSONAL_COLS} "
            "FROM kb.personal_doc_chunks c "
            "JOIN kb.personal_documents d ON d.id = c.doc_id "
            "WHERE c.user_id = ? AND d.status = 'done'"
        )
        try:
            raw = self._conn.execute(sql, [user_id]).fetchall()
        except Exception:
            return []
        if not raw:
            return []
        rows = [self._personal_row(r) for r in raw]
        bm25 = _BM25([r.content for r in rows])
        scores = bm25.scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(rows[i], None, float(scores[i])) for i in order]

    @staticmethod
    def _union_recall(
        vec_rows: list, bm25_rows: list
    ) -> list[tuple[_Row, Optional[float], Optional[float]]]:
        by_id: dict[int, list] = {}
        for row, sim, _ in vec_rows:
            by_id[row.id] = [row, sim, None]
        for row, _, bm in bm25_rows:
            if row.id in by_id:
                by_id[row.id][2] = bm  # 双路命中：补上 bm25 分
            else:
                by_id[row.id] = [row, None, bm]
        return list(by_id.values())

    # —— 检索主入口 —— #
    def search(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        effective_after: Optional[str] = None,
        vec_weight: float = 0.7,
        rerank: bool = False,
        user_id: Optional[int] = None,
        use_personal_docs: bool = False,
    ) -> list[Passage]:
        """双路召回（向量 + BM25，各自全库独立召回）取并集，min-max 归一化后加权融合重排。

        ``use_personal_docs=True`` 且给定 ``user_id`` 时，额外并入 08 该用户的个人文档块
        （同库同连接，评审 B2）；个人文档不参与 domain / effective_date 元数据过滤，
        因其无监管元数据，过滤条件仅对公共知识库生效。
        """
        beta = 3  # 扩召回倍数
        qvec = self._embed([query])[0]

        # 召回①：向量路，元数据预过滤后全库取 top_k*beta 候选
        vec_rows = self._duckdb_recall(qvec, top_k * beta, domain, effective_after)
        # 召回②：关键词路，全库（元数据预过滤后）独立召回 top_k*beta 候选
        #   BM25 是查询相关模型，对任意文本可打分，能补回向量漏召的块（评审 B1）
        bm25_rows = self._bm25_recall(query, top_k * beta, domain, effective_after)

        # 召回③④：个人文档（用户作用域），同样双路，与公共库候选统一融合
        if use_personal_docs and user_id is not None:
            vec_rows = vec_rows + self._personal_vector_recall(
                qvec, top_k * beta, user_id
            )
            bm25_rows = bm25_rows + self._personal_bm25_recall(
                query, top_k * beta, user_id
            )

        cand = self._union_recall(vec_rows, bm25_rows)
        if not cand:
            return []

        # 重排：缺失分填 0，min-max 归一化后加权融合
        nv = _normalize([c[1] if c[1] is not None else 0.0 for c in cand])
        nb = _normalize([c[2] if c[2] is not None else 0.0 for c in cand])
        fused = [vec_weight * a + (1 - vec_weight) * b for a, b in zip(nv, nb)]

        ranked = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
        if rerank and len(ranked) > top_k and self._reranker is not None:
            reranked = self._reranker(query, [cand[i][0] for i in ranked[: top_k * beta]], top_k)
            if reranked:
                ranked = reranked
        ranked = ranked[:top_k]

        return [
            Passage(
                content=cand[i][0].content,
                source=cand[i][0].title or cand[i][0].doc_id,
                score=round(fused[i], 4),
                chapter=cand[i][0].chapter,
                effective_date=cand[i][0].effective_date,
                title=cand[i][0].title,
                agency=cand[i][0].agency,
                doc_type=cand[i][0].doc_type,
                domain=cand[i][0].domain,
                version=cand[i][0].version,
                doc_id=cand[i][0].doc_id,
            )
            for i in ranked
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        *,
        user_id: Optional[int] = None,
        use_personal_docs: bool = False,
    ) -> list[Passage]:
        """02 既有契约：召回 + 重排，默认 top_k=3（可选并入个人文档，评审 B2）。"""
        return self.search(
            query,
            top_k=top_k,
            user_id=user_id,
            use_personal_docs=use_personal_docs,
        )

    def vector_search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        domain: Optional[str] = None,
        effective_after: Optional[str] = None,
    ) -> list[Passage]:
        """FR5：纯向量余弦检索（不融合 BM25，向量计算在 DuckDB 端完成）。"""
        sql = (
            "SELECT rowid AS id, doc_id, title, agency, doc_type, domain, version, "
            "chapter, content, effective_date, "
            "1 - list_cosine_distance(embedding::FLOAT[], ?::FLOAT[]) AS similarity "
            "FROM kb.chunks "
            "WHERE (? IS NULL OR list_contains(string_split(coalesce(domain, ''), ','), ?)) "
            "  AND (? IS NULL OR effective_date >= ?) "
            "ORDER BY similarity DESC LIMIT ?"
        )
        rows = self._conn.execute(
            sql, [query_vec, domain, domain, effective_after, effective_after, top_k]
        ).fetchall()
        return [
            Passage(
                content=r[8],
                source=r[2] or r[1],
                score=round(float(r[10]), 4),
                chapter=r[7] or "",
                effective_date=r[9] or "",
                title=r[2] or "",
                agency=r[3] or "",
                doc_type=r[4] or "",
                domain=r[5] or "",
                version=r[6] or "",
                doc_id=r[1] or "",
            )
            for r in rows
        ]

    # —— 生命周期 —— #
    def stats(self) -> dict:
        n_chunks, n_docs, max_dim = self._conn.execute(
            "SELECT count(*), count(DISTINCT doc_id), MAX(dim) FROM kb.chunks"
        ).fetchone()
        size = os.path.getsize(self._path) if os.path.exists(self._path) else 0
        row = self._conn.execute(
            "SELECT value FROM kb.kb_meta WHERE key = 'last_build_at'"
        ).fetchone()
        return {
            "doc_count": n_docs or 0,
            "chunk_count": n_chunks or 0,
            "dim": max_dim,
            "size_bytes": size,
            "last_build_at": row[0] if row else None,
        }

    def list_docs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc_id, title, doc_type, agency, domain, version, count(*) "
            "FROM kb.chunks GROUP BY doc_id, title, doc_type, agency, domain, version "
            "ORDER BY doc_id"
        ).fetchall()
        return [
            {
                "doc_id": r[0],
                "title": r[1] or "",
                "doc_type": r[2] or "",
                "agency": r[3] or "",
                "domain": r[4] or "",
                "version": r[5] or "",
                "chunk_count": r[6],
            }
            for r in rows
        ]

    def delete_doc(self, doc_id: str) -> int:
        before = self._conn.execute(
            "SELECT count(*) FROM kb.chunks WHERE doc_id = ?", [doc_id]
        ).fetchone()[0]
        self._conn.execute("DELETE FROM kb.chunks WHERE doc_id = ?", [doc_id])
        return before

    def rebuild(self, *, rescan: bool = False) -> int:
        """默认仅清空 chunks 表并返回删除块数；rescan=True 且配置了 kb_dir 时重新扫描入库。"""
        n = self._conn.execute("SELECT count(*) FROM kb.chunks").fetchone()[0]
        self._conn.execute("DELETE FROM kb.chunks")
        if rescan and self._kb_dir:
            self._rescan(self._kb_dir)
        return n

    def _rescan(self, kb_dir: str) -> int:
        """扫描 kb_dir 下文档重新 ingest（build 脚本与 rebuild(rescan=True) 共用）。"""
        import datetime

        total = 0
        for name in sorted(os.listdir(kb_dir)):
            if name.lower().endswith((".md", ".markdown", ".txt", ".html", ".htm")):
                total += self.ingest_document(os.path.join(kb_dir, name))
        now = datetime.datetime.now().isoformat()
        # 挂载的 SQLite 不支持 MERGE/ON CONFLICT，改用 DELETE + INSERT 幂等写入
        self._conn.execute("DELETE FROM kb.kb_meta WHERE key = 'last_build_at'")
        self._conn.execute(
            "INSERT INTO kb.kb_meta (key, value) VALUES ('last_build_at', ?)", [now]
        )
        return total

    def close(self) -> None:
        """关闭并逐出共享连接（同库其它使用者会在下次调用时重新挂载）。"""
        duckdb_conn.close(self._path, alias="kb")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 markdown YAML frontmatter，返回 (meta, body)。无 frontmatter 返回 ({}, text)。"""
    m = re.match(r"^\s*---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    fm_raw, body = m.group(1), m.group(2)
    meta: dict = {}
    for line in fm_raw.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body
