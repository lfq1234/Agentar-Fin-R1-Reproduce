"""04 扩展：RAG 本地文档管线（在 02 既有 `Passage` / `retrieve()` 之上扩展）。

保留 02 既有接口不变：
- `Passage`（增加可选字段 `chapter` / `effective_date`，默认值兼容既有调用）；
- `retrieve(query, top_k)`：02 的 `tools.py` 经此取检索片段，行为向后兼容
  （无索引时返回空列表，工具层照常输出「暂无检索结果」）。

新增能力（评审 G1 闭环，不新建包）：
- 接入/解析/切分（ingest_document / ingest_text + `_split_into_chunks`）；
- 向量化（经 01 的 `get_embedder()`，api/local 双模式）；
- 存储（本地 SQLite，chunks + 向量 + 元数据）；
- 混合检索（向量余弦 + BM25）+ 元数据过滤 + 重排（`search`）；
- 联网新闻占位（`news_search`，真实 API 为后续需求，提供可插拔 provider）。

定位：把本地金融文档转为可检索、可追溯的知识，供信息查询智能体（rag 角色）消费。

> 演进（06 知识库-DuckDB）：切分 / BM25 / `Passage` / 归一化 已下沉到纯 Python
> 共享模块 ``app.db.knowledge.chunking``，本文件仅 re-export 以保持 04 对外契约不变；
> SQLite 后端实现见 ``app.db.knowledge.sqlite_store``，DuckDB 后端见 ``app.db.knowledge.store``。
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from app.model import get_embedder

# 06 演进：以下纯 Python 符号下沉到 `app.db.knowledge.chunking`（零重依赖、无 torch），
# 此处 re-export，04 既有调用方（tools.py 等）无需改动即可取到 `Passage` 等。
from app.db.knowledge.chunking import (
    Passage,
    _Chunk,
    _split_into_chunks,
    _BM25,
    _normalize,
)


@dataclass
class NewsItem:
    """一条联网新闻摘要（04 外联新闻管线输出）。"""

    summary: str
    source: str = ""
    published_at: str = ""
    url: str = ""
    credibility: float = 0.0


# --------------------------------------------------------------------------- #
# 数学工具
# --------------------------------------------------------------------------- #


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# 检索器（核心）
# --------------------------------------------------------------------------- #


class Retriever:
    """本地 RAG 检索器：接入 → 切分 → 向量化 → 存储 → 混合检索。

    依赖 01 的 `get_embedder()` 完成向量化（api/local 双模式），懒加载，不在
    构造或导入时触发真实模型请求。无索引时 `search`/`retrieve` 返回空列表。
    """

    def __init__(self, db_path: str = ":memory:", embedder=None) -> None:
        self._db_path = db_path
        self._embedder = embedder  # 可注入（测试用）；为 None 时懒加载 01
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "doc_id TEXT, title TEXT, doc_type TEXT, agency TEXT,"
            "domain TEXT, chapter TEXT, effective_date TEXT,"
            "content TEXT, embedding TEXT)"
        )
        self._conn.commit()

    # —— embedder —— #
    def set_embedder(self, fn) -> None:
        """注入 embedding 函数（测试或自定义实现用）。"""
        self._embedder = fn

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()  # 01 扩展，懒加载
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._get_embedder().embed(texts)

    # —— ingest（接入 / 解析 / 切分 / 向量化 / 存储） —— #
    def ingest_text(
        self,
        text: str,
        *,
        doc_id: str = "doc",
        title: str = "",
        doc_type: str = "internal",
        agency: str = "",
        domain: str = "",
        effective_date: str = "",
        max_chars: int = 600,
        overlap: int = 80,
    ) -> int:
        """将一段文本切分、向量化并写入索引；返回写入块数。"""
        chunks = _split_into_chunks(text, max_chars=max_chars, overlap=overlap)
        if not chunks:
            return 0
        vecs = self._embed([c.text for c in chunks])
        cur = self._conn.cursor()
        for c, v in zip(chunks, vecs):
            cur.execute(
                "INSERT INTO chunks "
                "(doc_id,title,doc_type,agency,domain,chapter,effective_date,content,embedding) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    title,
                    doc_type,
                    agency,
                    domain,
                    c.chapter,
                    effective_date,
                    c.text,
                    json.dumps(v, ensure_ascii=False),
                ),
            )
        self._conn.commit()
        return len(chunks)

    def ingest_document(self, path: str, **meta) -> int:
        """按扩展名接入文档；txt/md/html 原生支持，二进制需可选依赖。"""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".txt", ".md", ".markdown"):
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            title = meta.pop("title", os.path.basename(path))
            doc_id = meta.pop("doc_id", os.path.basename(path))
            return self.ingest_text(text, doc_id=doc_id, title=title, **meta)
        if ext in (".html", ".htm"):
            with open(path, encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            text = re.sub(r"<script[\s\S]*?</script>", " ", raw)
            text = re.sub(r"<style[\s\S]*?</style>", " ", text)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&nbsp;", " ", text)
            text = re.sub(r"\s+", " ", text)
            title = meta.pop("title", os.path.basename(path))
            doc_id = meta.pop("doc_id", os.path.basename(path))
            return self.ingest_text(text, doc_id=doc_id, title=title, **meta)
        # 二进制格式需可选依赖；本期提供扩展点，缺失时给出清晰指引。
        raise NotImplementedError(
            f"暂不支持 {ext} 解析；PDF/DOCX/XLSX 需安装 pdfminer.six / python-docx / openpyxl，"
            f"本期 RAG 实现聚焦 txt/md/html，二进制解析为后续工程需求。"
        )

    # —— search（混合检索 + 过滤 + 重排） —— #
    def search(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        effective_after: Optional[str] = None,
        vec_weight: float = 0.7,
    ) -> list[Passage]:
        """混合检索：向量余弦 + BM25，按元数据过滤后融合重排。

        Args:
            query: 检索查询。
            top_k: 返回条数。
            domain: 领域过滤（逗号分隔多标签，命中其一即可）。
            effective_after: 生效日下界（ISO 字符串，优先现行有效）。
            vec_weight: 向量得分在融合中的权重（BM25 权重为 1 - vec_weight）。
        """
        # 列序：id, doc_id, title, domain, chapter, content, effective_date, embedding
        rows = self._conn.execute(
            "SELECT id,doc_id,title,domain,chapter,content,effective_date,embedding "
            "FROM chunks"
        ).fetchall()
        if not rows:
            return []

        # 元数据过滤
        cands = []
        for r in rows:
            dom = (r[3] or "").strip()
            eff = (r[6] or "").strip()
            if domain:
                tags = [x.strip() for x in dom.split(",") if x.strip()]
                if domain not in tags:
                    continue
            if effective_after and eff and eff < effective_after:
                continue
            cands.append(r)
        if not cands:
            return []

        contents = [r[5] for r in cands]
        qvec = self._embed([query])[0]
        bm25 = _BM25(contents)
        bm25_scores = bm25.scores(query)
        vec_scores = [_cosine(qvec, json.loads(r[7])) for r in cands]

        nv = _normalize(vec_scores)
        nb = _normalize(bm25_scores)
        fused = [vec_weight * a + (1 - vec_weight) * b for a, b in zip(nv, nb)]
        ranked = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)[:top_k]

        out: list[Passage] = []
        for i in ranked:
            r = cands[i]
            out.append(
                Passage(
                    content=r[5],
                    source=f"{r[2] or r[1]}",
                    score=round(fused[i], 4),
                    chapter=r[4] or "",
                    effective_date=r[6] or "",
                )
            )
        return out

    def retrieve(self, query: str, top_k: int = 3) -> list[Passage]:
        """02 既有接口：返回 `list[Passage]`；无索引时恒返回空列表。"""
        return self.search(query, top_k=top_k)


# --------------------------------------------------------------------------- #
# 模块级单例与便捷函数（02 的 tools.py 经 `retrieve` 取片段）
# --------------------------------------------------------------------------- #

_DEFAULT: Optional[Retriever] = None


def _default_retriever() -> Retriever:
    global _DEFAULT
    if _DEFAULT is None:
        db = os.environ.get("AGENTAR_RAG_DB", ":memory:")
        _DEFAULT = Retriever(db_path=db)
    return _DEFAULT


def retrieve(query: str, top_k: int = 3) -> list[Passage]:
    """模块级检索入口（02 既有），向后兼容。"""
    return _default_retriever().retrieve(query, top_k=top_k)


def ingest_text(text: str, **meta) -> int:
    """模块级文本接入入口。"""
    return _default_retriever().ingest_text(text, **meta)


def ingest_document(path: str, **meta) -> int:
    """模块级文档接入入口。"""
    return _default_retriever().ingest_document(path, **meta)


def news_search(
    query: str,
    provider: Optional[Callable[[str, int], list[NewsItem]]] = None,
    top_k: int = 5,
) -> list[NewsItem]:
    """联网新闻检索占位（真实 API 为后续需求）。

    provider: 可插拔的检索实现（输入 query, top_k，输出 NewsItem 列表）；
    未提供时返回空列表（信息查询智能体据此如实声明「未检索到相关近期信息」）。
    """
    if provider is None:
        return []
    return provider(query, top_k)
