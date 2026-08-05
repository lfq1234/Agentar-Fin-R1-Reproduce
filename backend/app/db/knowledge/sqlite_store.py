"""06-知识库：SQLite 文件存储后端（自包含，零 torch 依赖）。

职责（按用户约定「文件放 SQLite、向量检索放 DuckDB」）：
- 本模块是**文件存储的单一真相来源**：建表、写入（文档/块/embedding JSON）、
  生命周期（stats / list_docs / delete_doc / rebuild）全部落地到 SQLite 文件；
- ``kb.engine=sqlite`` 时，本类也是完整回落实现（向量检索用纯 Python 余弦 + BM25）；
- ``kb.engine=duckdb`` 时，``app.db.knowledge.store.DuckDBKnowledgeStore`` 通过 DuckDB
  ``ATTACH`` 挂载**同一个** SQLite 文件，仅在 SQL 端做向量检索（``list_cosine_distance``），
  写入与文件存储仍走本 schema。

> 顶层不 import torch（嵌入器惰性加载），满足「无 torch 单测」验收基线。
> 两后端共用本文件定义的 ``CHUNKS_DDL`` / ``KB_META_DDL``，避免 schema 漂移。
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from typing import Optional

from app.db.knowledge.chunking import (
    InconsistentDimensionError,
    Passage,
    _BM25,
    _normalize,
    _split_into_chunks,
)

# —— 唯一 schema 定义（DuckDB 引擎经 ATTACH 复用同一 DDL 建表） —— #
CHUNKS_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY,
    doc_id          TEXT,
    title           TEXT,
    doc_type        TEXT,
    agency          TEXT,
    domain          TEXT,
    chapter         TEXT,
    content         TEXT,
    effective_date  TEXT,
    version         TEXT,
    source_path     TEXT,
    embedding       TEXT,
    dim             INTEGER
)
"""

KB_META_DDL = """
CREATE TABLE IF NOT EXISTS kb_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _domain_tags(domain: str) -> list[str]:
    """逗号分隔的 domain 多标签规范化（过滤用，精确命中）。"""
    return [t.strip() for t in domain.split(",") if t.strip()]


class SQLiteKnowledgeStore:
    """SQLite 文件存储后端：自包含实现 06 统一契约。

    既作为 ``kb.engine=sqlite`` 的纯 SQLite 回落（向量检索用 Python 余弦），
    其 ``CHUNKS_DDL`` 也是 DuckDB 引擎挂载的同一 SQLite 文件的建表依据。
    """

    def __init__(self, db_path: str = ":memory:", *, embedder=None, kb_dir: Optional[str] = None) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(CHUNKS_DDL)
        self._conn.execute(KB_META_DDL)
        self._conn.commit()
        self._embedder = embedder
        self._kb_dir = kb_dir
        # 维度：从库内已有数据 seed（自适应），保证跨进程一致性
        row = self._conn.execute("SELECT MAX(dim) FROM chunks").fetchone()
        self._dim = row[0] if row and row[0] is not None else None

    # —— embedder —— #
    def set_embedder(self, fn) -> None:
        self._embedder = fn

    def _get_embedder(self):
        if self._embedder is None:
            from app.model import get_embedder  # 惰性：避免导入即拉起 torch

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
        """切分 + 向量化 + 写入 SQLite（单 doc 事务包裹，评审 I3）；返回写入块数。"""
        chunks = _split_into_chunks(text, max_chars=max_chars, overlap=overlap)
        if not chunks:
            return 0
        vecs = self._embed([c.text for c in chunks])

        # 维度自适应校验
        for v in vecs:
            if self._dim is None:
                self._dim = len(v)
            if self._dim != len(v):
                raise InconsistentDimensionError(
                    f"embedding 维度不一致：库内 {self._dim}，当前 {len(v)}；"
                    f"切换 embedder 需 rebuild 知识库。"
                )

        dom = ",".join(_domain_tags(domain))
        try:
            self._conn.execute("BEGIN")
            # 增量去重：同 doc_id 视为新版本，先删旧再写（FR2）
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", [doc_id])
            for c, v in zip(chunks, vecs):
                self._conn.execute(
                    "INSERT INTO chunks "
                    "(doc_id,title,doc_type,agency,domain,chapter,content,"
                    "effective_date,version,source_path,embedding,dim) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        doc_id, title, doc_type, agency, dom, c.chapter, c.text,
                        effective_date, version, source_path,
                        json.dumps(v, ensure_ascii=False), len(v),
                    ],
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
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

    # —— 候选集（元数据预过滤） —— #
    def _rows_filtered(self, domain: Optional[str], effective_after: Optional[str]):
        rows = self._conn.execute(
            "SELECT id,doc_id,title,agency,doc_type,domain,version,chapter,"
            "content,effective_date,embedding FROM chunks"
        ).fetchall()
        out = []
        for r in rows:
            dom = (r[5] or "").strip()
            eff = (r[9] or "").strip()
            if domain:
                tags = [x.strip() for x in dom.split(",") if x.strip()]
                if domain not in tags:
                    continue
            if effective_after and eff and eff < effective_after:
                continue
            out.append(r)
        return out

    @staticmethod
    def _to_passage(r, score: float) -> Passage:
        return Passage(
            content=r[8],
            source=r[2] or r[1],
            score=round(score, 4),
            chapter=r[7] or "",
            effective_date=r[9] or "",
            title=r[2] or "",
            agency=r[3] or "",
            doc_type=r[4] or "",
            domain=r[5] or "",
            version=r[6] or "",
            doc_id=r[1] or "",
        )

    # —— FR5：纯向量余弦检索（回落实现） —— #
    def vector_search(
        self,
        query_vec: list[float],
        top_k: int = 5,
        domain: Optional[str] = None,
        effective_after: Optional[str] = None,
    ) -> list[Passage]:
        cands = self._rows_filtered(domain, effective_after)
        if not cands:
            return []
        scored = [(r, _cosine(query_vec, json.loads(r[10]))) for r in cands]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._to_passage(r, sim) for r, sim in scored[:top_k]]

    # —— 混合检索（向量 + BM25 融合重排） —— #
    def search(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        effective_after: Optional[str] = None,
        vec_weight: float = 0.7,
    ) -> list[Passage]:
        cands = self._rows_filtered(domain, effective_after)
        if not cands:
            return []
        contents = [r[8] for r in cands]
        qvec = self._embed([query])[0]
        bm25 = _BM25(contents)
        bm25_scores = bm25.scores(query)
        vec_scores = [_cosine(qvec, json.loads(r[10])) for r in cands]

        nv = _normalize(vec_scores)
        nb = _normalize(bm25_scores)
        fused = [vec_weight * a + (1 - vec_weight) * b for a, b in zip(nv, nb)]
        ranked = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)[:top_k]

        return [self._to_passage(cands[i], fused[i]) for i in ranked]

    def retrieve(self, query: str, top_k: int = 3) -> list[Passage]:
        return self.search(query, top_k=top_k)

    # —— 生命周期 —— #
    def stats(self) -> dict:
        n_chunks, n_docs, max_dim = self._conn.execute(
            "SELECT count(*), count(DISTINCT doc_id), MAX(dim) FROM chunks"
        ).fetchone()
        size = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0
        row = self._conn.execute(
            "SELECT value FROM kb_meta WHERE key = 'last_build_at'"
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
            "FROM chunks GROUP BY doc_id, title, doc_type, agency, domain, version "
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
            "SELECT count(*) FROM chunks WHERE doc_id = ?", [doc_id]
        ).fetchone()[0]
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", [doc_id])
        return before

    def rebuild(self, *, rescan: bool = False) -> int:
        """默认仅清空 chunks 表并返回删除块数；rescan=True 且配置了 kb_dir 时重新扫描入库。"""
        n = self._conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        self._conn.execute("DELETE FROM chunks")
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
        self._conn.execute(
            "INSERT INTO kb_meta (key, value) VALUES ('last_build_at', ?) "
            "ON CONFLICT (key) DO UPDATE SET value = ?",
            [now, now],
        )
        return total

    def close(self) -> None:
        self._conn.close()


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
